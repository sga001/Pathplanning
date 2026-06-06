from __future__ import annotations

import argparse
import dataclasses
import filecmp
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import irsim
import numpy as np
import yaml


DEFAULT_TIMEOUT_S = 120.0
LIDAR_BEAM_COUNT = 360
ACTION_SHAPE = (2, 1)
RENDER_PAUSE_SECONDS = 0.05


class ArenaConfigError(ValueError):
    """Raised at Arena.__init__ for malformed config (e.g. lidar beam count mismatch)."""


class ArenaRuntimeError(RuntimeError):
    """Raised mid-episode for irsim contract violations (e.g. lidar dict missing 'ranges')."""


# Bootstrap repo root on sys.path so the `from arena.dynamic import ...` below
# resolves whether this file is run as `python arena/arena.py` (script-mode puts
# arena/ on sys.path, not the repo root) or as `python -m arena.arena` / via the
# runner (repo root already on sys.path). Mirrors runners/run_episode.py:39-43.
import sys as _sys
from pathlib import Path as _Path
_repo_root = str(_Path(__file__).resolve().parent.parent)
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

# NOTE: imported AFTER ArenaRuntimeError is defined so the circular dependency
# (arena.dynamic imports ArenaRuntimeError from arena.arena) resolves cleanly.
from arena.dynamic import DynamicObstacleState, TrafficSpawner  # noqa: E402


@dataclass(frozen=True)
class EpisodeInfo:
    sim_time: float
    step_idx: int
    crashed: bool
    timed_out: bool
    reached_goal: bool
    distance_to_goal: float
    wallclock_per_step: float
    dynamic_obstacle_count: int
    lidar_status: str
    dynamic_obstacles_sha256: str | None


class Arena:
    """50x50 arena wrapping irsim. Static-only by default; pass traffic=True for Phase 2 crossing traffic."""

    def __init__(
        self,
        yaml_path: str | Path,
        seed: int,
        render: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        traffic: bool = False,
    ) -> None:
        self._yaml_path = Path(yaml_path)
        self._render = bool(render)
        self._timeout_s = float(timeout_s)
        self._master_seed = int(seed)
        self._traffic = bool(traffic)

        # With traffic on, every dynamic obstacle (omni, no behavior) makes irsim log
        # a per-tick WARNING ("Behavior not defined ..."), ~20 lines/tick that would
        # flood the runner output. Raise the irsim log level to ERROR for traffic runs
        # (collision/arrival are read from flags, not these logs); Phase 0/1 runs keep
        # the default level so their logging is unchanged.
        log_level = "ERROR" if self._traffic else "INFO"
        self._env = irsim.make(
            str(self._yaml_path), display=self._render, log_level=log_level
        )
        self._robot = self._env.robot_list[0]
        self._dt = float(self._env.step_time)
        goal = self._robot.goal
        if goal is None:
            raise ArenaConfigError("YAML robot has no goal pose")
        self._goal_xy = goal[:2, 0].astype(np.float64)

        scan = self._robot.get_lidar_scan()
        if not scan:
            raise ArenaConfigError("YAML robot has no working lidar2d sensor")
        if "ranges" not in scan:
            raise ArenaConfigError(
                f"YAML lidar2d scan dict missing 'ranges' key: keys={list(scan.keys())}"
            )
        beam_count = len(scan["ranges"])
        if beam_count != LIDAR_BEAM_COUNT:
            raise ArenaConfigError(
                f"lidar scan returned {beam_count} beams, expected {LIDAR_BEAM_COUNT}"
            )

        # traffic first, motion second — Phase 2 spawner consumes in this order
        ss = np.random.SeedSequence(self._master_seed)
        traffic_seed, motion_seed = ss.spawn(2)
        self._traffic_rng = np.random.default_rng(traffic_seed)
        self._motion_rng = np.random.default_rng(motion_seed)

        # Cache the WorldModel ONCE for the spawner construction (and for reuse on reset()).
        # Lazy-import manual_astar (mirrors TC10 pattern) — keeps arena import-time cheap
        # and avoids cycles if manual_astar grows imports.
        self._world_model: Any | None = None
        if self._traffic:
            import sys as _sys
            _repo_root = str(Path(__file__).resolve().parent.parent)
            if _repo_root not in _sys.path:
                _sys.path.insert(0, _repo_root)
            from manual_astar import load_world  # type: ignore[import-not-found]

            self._world_model = load_world(str(self._yaml_path))
            self._spawner: TrafficSpawner | None = TrafficSpawner(
                env=self._env,
                robot=self._robot,
                traffic_rng=self._traffic_rng,
                motion_rng=self._motion_rng,
                dt=self._dt,
                arena_w=float(self._world_model.width),
                arena_h=float(self._world_model.height),
                static_obstacles=self._world_model.obstacles,
            )
        else:
            self._spawner = None

        # Pre-reset snapshot caches: per AC13, initial_dynamic_snapshot must return ()
        # and EpisodeInfo.dynamic_obstacles_sha256 must be None until reset() runs.
        # _initial_snapshot is captured ONCE at reset-time (the t=0 view planners get
        # via the public property); _last_snapshot tracks the per-tick state for
        # EpisodeInfo + sha256 bookkeeping inside step().
        self._initial_snapshot: tuple[DynamicObstacleState, ...] = ()
        self._last_snapshot: tuple[DynamicObstacleState, ...] = ()
        self._last_sha256: str | None = None

        self._step_idx = 0
        self._done = False
        self._closed = False
        self._reset_called = False

    def reset(self) -> tuple[np.ndarray, np.ndarray, EpisodeInfo]:
        if self._closed:
            raise RuntimeError("Arena is closed")

        # Step 1: irsim reset (resets all current objects to _init_state, runs warm-up step)
        self._env.reset()

        # Step 2: re-derive RNGs deterministically from master seed (mirrors __init__
        # exactly so reset() is byte-equivalent to fresh construct + reset).
        ss = np.random.SeedSequence(self._master_seed)
        traffic_seed, motion_seed = ss.spawn(2)
        self._traffic_rng = np.random.default_rng(traffic_seed)
        self._motion_rng = np.random.default_rng(motion_seed)

        # Step 3: clear the PRIOR episode's dynamic obstacles. env.reset() resets their
        # POSE but does not remove them; spawning again without clearing would DOUBLE
        # the population. The spawner owns its own teardown (delete-all + RNG rebind) —
        # the cached point-distance callable and static-obstacle list are preserved,
        # and its _next_idx keeps climbing so obstacle names never collide.
        if self._spawner is not None:
            self._spawner.reset(self._traffic_rng, self._motion_rng)

        # Step 4: spawn fresh population (if traffic enabled). The initial snapshot
        # is pinned here and exposed via the initial_dynamic_snapshot property for the
        # full episode — _last_snapshot is overwritten on every step() but the t=0
        # view planners depend on must never drift.
        if self._spawner is not None:
            self._initial_snapshot = self._spawner.initialize()
            # env.reset()'s warm-up sensed the lidar BEFORE these obstacles existed, so
            # re-sense now: lidar0 must be consistent with the snapshot/sha the planner
            # receives for the same t=0 (reactive planners consume lidar0).
            self._robot.sensor_step()
            self._last_snapshot = self._initial_snapshot
            self._last_sha256 = self._spawner.state_sha256(self._last_snapshot)
        else:
            self._initial_snapshot = ()
            self._last_snapshot = ()
            self._last_sha256 = None

        # Step 5: defensive flag re-clear — irsim's reset() warm-up step may set these
        # against the just-reset pose if it overlaps an obstacle (Phase 0 T0 note).
        self._robot.arrive_flag = False
        self._robot.collision_flag = False

        # Step 6: counter reset
        self._step_idx = 0
        self._done = False
        self._reset_called = True

        # Step 7: build initial state + lidar + EpisodeInfo
        state = self._robot.state[:, 0].astype(np.float64)
        lidar, lidar_status = self._extract_lidar()

        info = EpisodeInfo(
            sim_time=0.0,
            step_idx=0,
            crashed=False,
            timed_out=False,
            reached_goal=False,
            distance_to_goal=float(np.linalg.norm(state[:2] - self._goal_xy)),
            wallclock_per_step=0.0,
            dynamic_obstacle_count=len(self._last_snapshot),
            lidar_status=lidar_status,
            dynamic_obstacles_sha256=self._last_sha256,
        )
        return state, lidar, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, EpisodeInfo]:
        if self._done:
            raise RuntimeError("Episode is done; call reset() first.")
        if self._closed:
            raise RuntimeError("Arena is closed")
        if not self._reset_called:
            raise RuntimeError("reset() must be called before step()")

        if not isinstance(action, np.ndarray):
            raise ValueError(
                f"action must be np.ndarray, got {type(action).__name__}"
            )
        if action.shape != ACTION_SHAPE:
            raise ValueError(
                f"action shape must be {ACTION_SHAPE}, got {action.shape}"
            )
        if not np.issubdtype(action.dtype, np.floating):
            raise ValueError(f"action dtype must be float, got {action.dtype}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action contains NaN or Inf")

        # Advance dynamic obstacles BEFORE env.step() so the lidar inside env.step()
        # samples post-move obstacle positions on the same tick.
        if self._spawner is not None:
            self._last_snapshot = self._spawner.step()
            self._last_sha256 = self._spawner.state_sha256(self._last_snapshot)

        # Snapshot flags BEFORE step: irsim's check_*_status overwrite them per tick
        # (see object_base.py:531-532), so harness-injected flags would be lost otherwise.
        pre_crashed = bool(getattr(self._robot, "collision_flag", False))
        pre_reached = bool(getattr(self._robot, "arrive_flag", False))

        start = time.perf_counter()
        self._env.step([action])
        wallclock = time.perf_counter() - start

        # Drive irsim's repaint loop when render mode is on. Without this, the window
        # never updates between steps and the user only sees the final frame.
        # Excluded from wallclock_per_step on purpose.
        if self._render:
            self._env.render(RENDER_PAUSE_SECONDS)

        self._step_idx += 1

        state = self._robot.state[:, 0].astype(np.float64)
        lidar, lidar_status = self._extract_lidar()

        crashed = pre_crashed or bool(getattr(self._robot, "collision_flag", False))
        reached_goal = pre_reached or bool(getattr(self._robot, "arrive_flag", False))
        sim_time = self._step_idx * self._dt
        timed_out = sim_time >= self._timeout_s
        distance_to_goal = float(np.linalg.norm(state[:2] - self._goal_xy))

        info = EpisodeInfo(
            sim_time=sim_time,
            step_idx=self._step_idx,
            crashed=crashed,
            timed_out=timed_out,
            reached_goal=reached_goal,
            distance_to_goal=distance_to_goal,
            wallclock_per_step=wallclock,
            dynamic_obstacle_count=len(self._last_snapshot),
            lidar_status=lidar_status,
            dynamic_obstacles_sha256=self._last_sha256,
        )

        done = crashed or timed_out or reached_goal
        self._done = bool(done)
        return state, lidar, self._done, info

    def _extract_lidar(self) -> tuple[np.ndarray, str]:
        scan = self._robot.get_lidar_scan()
        if not scan:
            return (
                np.full((LIDAR_BEAM_COUNT,), np.nan, dtype=np.float64),
                "missing",
            )
        if "ranges" not in scan:
            raise ArenaRuntimeError(
                f"irsim lidar returned a non-falsy scan without 'ranges' key: "
                f"keys={list(scan.keys())}"
            )
        ranges = np.asarray(scan["ranges"], dtype=np.float64)
        if ranges.shape != (LIDAR_BEAM_COUNT,):
            raise ArenaRuntimeError(
                f"irsim lidar returned ranges of shape {ranges.shape}, "
                f"expected ({LIDAR_BEAM_COUNT},)"
            )
        range_max = float(scan.get("range_max", np.inf))
        ranges = np.where(
            np.isfinite(ranges) & (ranges < range_max), ranges, np.nan
        )
        return ranges, "ok"

    @property
    def initial_dynamic_snapshot(self) -> tuple[DynamicObstacleState, ...]:
        """Snapshot of dynamic obstacles at t=0 of the current episode.

        Phase 0/1: always (). Phase 2 with traffic=True: tuple of 20 DynamicObstacleState
        entries captured by TrafficSpawner.initialize() at reset-time. This is the
        t=0 view Mission.md guarantees to planners — it does NOT update on subsequent
        step() calls. Mid-episode dynamic state is not exposed in Phase 2; Phase 6
        replanners that need the live set will query the spawner separately.
        """
        return self._initial_snapshot

    def close(self) -> None:
        if self._closed:
            return
        if self._spawner is not None:
            self._spawner.close()
        self._env.end()
        self._closed = True


# ---------------------------------------------------------------------------
# TC1..TC12 — executable verification suite (run via `--check` from __main__).
# Each TC builds its own Arena, runs its assertions, and always calls close()
# in a finally block. Raise AssertionError on failure with a clear message.
# ---------------------------------------------------------------------------


EXPECTED_EPISODE_INFO_FIELDS = (
    "sim_time",
    "step_idx",
    "crashed",
    "timed_out",
    "reached_goal",
    "distance_to_goal",
    "wallclock_per_step",
    "dynamic_obstacle_count",
    "lidar_status",
    "dynamic_obstacles_sha256",
)


def tc1(yaml_path: str, seed: int) -> None:
    """Construct an Arena and close it cleanly."""
    arena = Arena(yaml_path, seed)
    arena.close()


def tc2(yaml_path: str, seed: int) -> None:
    """Reset returns correctly shaped state/lidar and a fully populated EpisodeInfo."""
    arena = Arena(yaml_path, seed)
    try:
        state, lidar, info = arena.reset()

        assert isinstance(state, np.ndarray), f"state must be np.ndarray, got {type(state).__name__}"
        assert state.shape == (3,), f"state.shape must be (3,), got {state.shape}"

        assert isinstance(lidar, np.ndarray), f"lidar must be np.ndarray, got {type(lidar).__name__}"
        assert lidar.shape == (LIDAR_BEAM_COUNT,), (
            f"lidar.shape must be ({LIDAR_BEAM_COUNT},), got {lidar.shape}"
        )
        assert lidar.dtype == np.float64, f"lidar.dtype must be float64, got {lidar.dtype}"

        assert isinstance(info, EpisodeInfo), (
            f"info must be an EpisodeInfo, got {type(info).__name__}"
        )
        field_names = tuple(f.name for f in dataclasses.fields(info))
        assert field_names == EXPECTED_EPISODE_INFO_FIELDS, (
            f"EpisodeInfo fields mismatch: got {field_names}, "
            f"expected {EXPECTED_EPISODE_INFO_FIELDS}"
        )
        assert info.lidar_status == "ok", (
            f"info.lidar_status must be 'ok' on a healthy reset, got {info.lidar_status!r}"
        )
    finally:
        arena.close()


def tc2b(yaml_path: str, seed: int) -> None:
    """Missing lidar tick: monkeypatch get_lidar_scan to return None and step once."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        original_scan = arena._robot.get_lidar_scan
        arena._robot.get_lidar_scan = lambda: None
        try:
            _, lidar, _, info = arena.step(np.array([[0.0], [0.0]], dtype=float))
            assert lidar.shape == (LIDAR_BEAM_COUNT,), (
                f"lidar.shape must be ({LIDAR_BEAM_COUNT},), got {lidar.shape}"
            )
            assert np.all(np.isnan(lidar)), "lidar must be all NaN when scan is missing"
            assert info.lidar_status == "missing", (
                f"info.lidar_status must be 'missing', got {info.lidar_status!r}"
            )
        finally:
            arena._robot.get_lidar_scan = original_scan
    finally:
        arena.close()


def tc3(yaml_path: str, seed: int) -> None:
    """One zero-action step: not done, step_idx advances, sim_time increments by dt."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        _, _, done, info = arena.step(np.array([[0.0], [0.0]], dtype=float))
        assert done is False, f"done must be False after one zero-action step, got {done}"
        assert info.step_idx == 1, f"info.step_idx must be 1, got {info.step_idx}"
        assert abs(info.sim_time - arena._dt) < 1e-9, (
            f"info.sim_time must equal dt={arena._dt}, got {info.sim_time}"
        )
    finally:
        arena.close()


def tc4(yaml_path: str, seed: int) -> None:
    """Deliberate crash: v=1.0, w=0.3 curves into pillar (5, 5) within 200 steps."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        action = np.array([[1.0], [0.3]], dtype=float)
        max_steps = 200
        done = False
        info: EpisodeInfo | None = None
        for _ in range(max_steps):
            _, _, done, info = arena.step(action)
            if done:
                break
        assert done, (
            f"Episode did not terminate within {max_steps} steps; "
            f"final info={info}"
        )
        assert info is not None and info.crashed, (
            f"Expected info.crashed == True after curved drive, got info={info}"
        )
    finally:
        arena.close()


def tc5(yaml_path: str, seed: int) -> None:
    """Standing still must trigger timeout once sim_time >= timeout_s."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        action = np.array([[0.0], [0.0]], dtype=float)
        max_iters = int(DEFAULT_TIMEOUT_S / arena._dt) + 5
        done = False
        info: EpisodeInfo | None = None
        for _ in range(max_iters):
            _, _, done, info = arena.step(action)
            if done:
                break
        assert done, (
            f"Episode did not terminate within {max_iters} zero-action steps; "
            f"final info={info}"
        )
        assert info is not None and info.timed_out, (
            f"Expected info.timed_out == True after standing still, got info={info}"
        )
    finally:
        arena.close()


def tc6(yaml_path: str, seed: int) -> None:
    """Calling step() after done == True must raise RuntimeError."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        arena._done = True
        try:
            arena.step(np.array([[0.0], [0.0]], dtype=float))
        except RuntimeError:
            return
        raise AssertionError("step() after done must raise RuntimeError, but it did not")
    finally:
        arena.close()


def tc7(yaml_path: str, seed: int) -> None:
    """reset() after a finished episode clears sticky state and zeroes counters."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        crash_action = np.array([[1.0], [0.3]], dtype=float)
        done = False
        for _ in range(200):
            _, _, done, _ = arena.step(crash_action)
            if done:
                break
        assert done, "Setup for TC7 failed: episode did not terminate via crash drive"

        _, _, info = arena.reset()
        assert info.sim_time == 0.0, f"info.sim_time must be 0.0 after reset, got {info.sim_time}"
        assert info.step_idx == 0, f"info.step_idx must be 0 after reset, got {info.step_idx}"
        assert info.crashed is False, f"info.crashed must be False after reset, got {info.crashed}"
        assert info.timed_out is False, (
            f"info.timed_out must be False after reset, got {info.timed_out}"
        )
        assert info.reached_goal is False, (
            f"info.reached_goal must be False after reset, got {info.reached_goal}"
        )
        assert arena._done is False, (
            f"Arena._done must be cleared after reset, got {arena._done}"
        )
    finally:
        arena.close()


def tc8(yaml_path: str, seed: int) -> None:
    """Injecting robot.arrive_flag=True before a zero step must mark reached_goal/done."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        arena._robot.arrive_flag = True
        _, _, done, info = arena.step(np.array([[0.0], [0.0]], dtype=float))
        assert done is True, f"done must be True after arrive_flag injection, got {done}"
        assert info.reached_goal is True, (
            f"info.reached_goal must be True after arrive_flag injection, got {info.reached_goal}"
        )
    finally:
        arena.close()


def tc9(yaml_path: str, seed: int) -> None:
    """All malformed actions must raise ValueError."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        bad_actions: list[tuple[str, Any]] = [
            ("list-not-ndarray", [0.0, 0.0]),
            ("wrong-shape-(1,2)", np.array([[0.0, 0.0]], dtype=float)),
            ("int-dtype", np.array([[0], [0]], dtype=int)),
            ("contains-NaN", np.array([[np.nan], [0.0]], dtype=float)),
            ("contains-Inf", np.array([[np.inf], [0.0]], dtype=float)),
        ]
        failures: list[str] = []
        for label, bad in bad_actions:
            try:
                arena.step(bad)
            except ValueError:
                continue
            except Exception as exc:
                failures.append(
                    f"{label}: expected ValueError, got {type(exc).__name__}: {exc}"
                )
                continue
            failures.append(f"{label}: expected ValueError, but step() returned normally")
        if failures:
            raise AssertionError("; ".join(failures))
    finally:
        arena.close()


def tc10(yaml_path: str, seed: int) -> None:  # noqa: ARG001 (seed unused — planner is deterministic in yaml)
    """manual_astar.py must accept the world: load, inflate, validate start/goal unblocked."""
    # Local import keeps Arena import-time cheap and avoids cycles if manual_astar grows imports.
    # manual_astar.py lives at the repo root; ensure it's importable when arena.py is invoked
    # from any cwd (e.g. `python arena/arena.py ...` puts `arena/` on sys.path, not the root).
    import sys
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from manual_astar import (  # type: ignore[import-not-found]
        GRID_RESOLUTION,
        SAFETY_MARGIN,
        build_occupancy_grid,
        load_world,
        validate_start_and_goal,
    )

    world = load_world(yaml_path)
    grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)
    validate_start_and_goal(world, grid)


def tc11(yaml_path: str, seed: int) -> None:  # noqa: ARG001
    """YAML schema sanity: world size, start/goal poses, and obstacle composition."""
    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    assert data["world"]["width"] == 50, (
        f"world.width must be 50, got {data['world']['width']}"
    )
    assert data["world"]["height"] == 50, (
        f"world.height must be 50, got {data['world']['height']}"
    )
    assert data["robot"]["state"] == [2, 2, 0], (
        f"robot.state must be [2, 2, 0], got {data['robot']['state']}"
    )
    assert data["robot"]["goal"] == [48, 48, 0], (
        f"robot.goal must be [48, 48, 0], got {data['robot']['goal']}"
    )

    obstacles = data["obstacle"]
    assert len(obstacles) == 14, f"expected 14 obstacles, got {len(obstacles)}"

    rect_count = sum(1 for o in obstacles if o["shape"]["name"] == "rectangle")
    circle_count = sum(1 for o in obstacles if o["shape"]["name"] == "circle")
    assert rect_count == 2, f"expected exactly 2 rectangle obstacles, got {rect_count}"
    assert circle_count == 12, f"expected exactly 12 circle obstacles, got {circle_count}"


def tc12(yaml_path: str, seed: int) -> None:  # noqa: ARG001
    """A YAML whose lidar2d.number != 360 must trigger ArenaConfigError at construction."""
    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    # Mutate beam count so Arena.__init__'s validation rejects it.
    data["robot"]["sensors"][0]["number"] = 180

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    try:
        yaml.safe_dump(data, tmp)
        tmp.close()
        tmp_path = tmp.name
        try:
            Arena(tmp_path, seed=0)
        except ArenaConfigError:
            return
        except Exception as exc:
            raise AssertionError(
                f"expected ArenaConfigError, got {type(exc).__name__}: {exc}"
            )
        raise AssertionError(
            "expected ArenaConfigError when lidar2d.number != 360, but construction succeeded"
        )
    finally:
        try:
            tmp.close()
        except Exception:
            pass
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


TC13_MAX_STEPS = 100


def tc13(yaml_path: str, seed: int) -> None:
    """Teleport robot under Wall B, drive forward, and assert crash within budget."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()

        # ObjectBase.state is read-only; set_state also refreshes geometry for collision checks.
        arena._robot.set_state([20.0, 19.0, np.pi / 2], init=False)

        arena._robot.collision_flag = False
        arena._robot.arrive_flag = False

        action = np.array([[1.0], [0.0]], dtype=float)
        done = False
        info: EpisodeInfo | None = None
        for _ in range(TC13_MAX_STEPS):
            _, _, done, info = arena.step(action)
            if done and info.crashed:
                break

        assert done and info is not None and info.crashed, (
            f"TC13 did not crash within {TC13_MAX_STEPS} steps; final info={info}"
        )
    finally:
        arena.close()


# ---------------------------------------------------------------------------
# TC14..TC16 — runner integration checks. These subprocess-invoke
# `python -m runners.run_episode`, then validate the resulting metrics JSON
# and trace JSONL artifacts under a tempdir. Subprocess cwd is the repo root
# (parent of arena/) so `runners.run_episode` resolves correctly and relative
# --world paths like `arena/arena_v1.yaml` resolve from the same anchor.
# ---------------------------------------------------------------------------


TC14_TRACE_REQUIRED_KEYS = frozenset(
    {"action", "crashed", "done", "lidar_sha256", "reached_goal", "state", "step"}
)
TC14_METRICS_REQUIRED_KEYS = frozenset(
    {
        "time_to_goal",
        "crashed",
        "timed_out",
        "path_length",
        "mean_speed",
        "wallclock_per_step",
        "planner_error",
    }
)
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def tc14(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses fixed internal seed for determinism
    """Full a_star_once drive through run_episode + trace-line schema audit."""
    repo_root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as td:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "runners.run_episode",
                "--algorithm",
                "a_star_once",
                "--seed",
                "42",
                "--world",
                yaml_path,
                "--results-dir",
                td,
                "--no-traffic",
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"TC14 runner exit code {result.returncode}; "
            f"stderr={result.stderr[-500:]}"
        )

        json_path = Path(td) / "arena_v1" / "a_star_once" / "42.json"
        jsonl_path = Path(td) / "arena_v1" / "a_star_once" / "42.trace.jsonl"
        assert json_path.exists(), f"TC14: metrics JSON missing at {json_path}"
        assert jsonl_path.exists(), f"TC14: trace JSONL missing at {jsonl_path}"

        metrics = json.loads(json_path.read_text(encoding="utf-8"))
        # Lazy-import speed bounds from manual_astar (mirrors tc10's sys.path pattern).
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        from manual_astar import (  # type: ignore[import-not-found]
            MAX_LINEAR_SPEED,
            MIN_LINEAR_SPEED,
        )

        assert set(metrics) == TC14_METRICS_REQUIRED_KEYS, (
            f"TC14 metrics keys mismatch: got {set(metrics)}, "
            f"expected {set(TC14_METRICS_REQUIRED_KEYS)}"
        )
        assert metrics["planner_error"] is None, f"TC14 planner_error not None: {metrics}"
        assert metrics["crashed"] is False, f"TC14 crashed: {metrics}"
        assert metrics["timed_out"] is False, f"TC14 timed_out: {metrics}"
        assert metrics["time_to_goal"] is not None, f"TC14 time_to_goal is None: {metrics}"
        assert 50.0 < metrics["time_to_goal"] < 120.0, (
            f"TC14 time_to_goal out of range (50, 120): {metrics}"
        )
        assert metrics["path_length"] > 64.0, f"TC14 path_length too short: {metrics}"
        assert MIN_LINEAR_SPEED <= metrics["mean_speed"] <= MAX_LINEAR_SPEED, (
            f"TC14 mean_speed out of [{MIN_LINEAR_SPEED}, {MAX_LINEAR_SPEED}]: {metrics}"
        )
        assert metrics["mean_speed"] > 0.5, f"TC14 mean_speed too low: {metrics}"

        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) > 1, f"TC14 trace too short: {len(lines)} lines"
        for idx, line in enumerate(lines):
            row = json.loads(line)
            assert set(row) == TC14_TRACE_REQUIRED_KEYS, (
                f"TC14 trace line {idx} keys mismatch: got {set(row)}, "
                f"expected {set(TC14_TRACE_REQUIRED_KEYS)}"
            )
            assert isinstance(row["step"], int), (
                f"TC14 line {idx} step type: {type(row['step']).__name__}"
            )
            assert isinstance(row["state"], list) and len(row["state"]) == 3, (
                f"TC14 line {idx} state shape: {row['state']!r}"
            )
            assert all(isinstance(x, (int, float)) for x in row["state"]), (
                f"TC14 line {idx} state element types: {row['state']!r}"
            )
            assert isinstance(row["action"], list) and len(row["action"]) == 2, (
                f"TC14 line {idx} action shape: {row['action']!r}"
            )
            assert all(isinstance(x, (int, float)) for x in row["action"]), (
                f"TC14 line {idx} action element types: {row['action']!r}"
            )
            assert isinstance(row["lidar_sha256"], str) and _HEX64_RE.match(
                row["lidar_sha256"]
            ), f"TC14 line {idx} lidar_sha256: {row['lidar_sha256']!r}"
            assert isinstance(row["crashed"], bool), (
                f"TC14 line {idx} crashed type: {type(row['crashed']).__name__}"
            )
            assert isinstance(row["reached_goal"], bool), (
                f"TC14 line {idx} reached_goal type: {type(row['reached_goal']).__name__}"
            )
            assert isinstance(row["done"], bool), (
                f"TC14 line {idx} done type: {type(row['done']).__name__}"
            )

        first = json.loads(lines[0])
        assert first["step"] == 0, f"TC14 first line step != 0: {first}"
        assert first["state"] == [2.0, 2.0, 0.0], (
            f"TC14 first line state != [2.0, 2.0, 0.0]: {first}"
        )
        assert first["action"] == [0.0, 0.0], (
            f"TC14 first line action != [0.0, 0.0]: {first}"
        )
        assert first["done"] is False and first["reached_goal"] is False, (
            f"TC14 first line done/reached_goal flags: {first}"
        )

        last = json.loads(lines[-1])
        assert last["done"] is True, f"TC14 last line done != True: {last}"
        assert last["reached_goal"] is True, (
            f"TC14 last line reached_goal != True: {last}"
        )


def tc15(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """Determinism: two same-seed subprocess runs produce byte-identical trace JSONL."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    runner_args = [
        sys.executable,
        "-m",
        "runners.run_episode",
        "--algorithm",
        "a_star_once",
        "--seed",
        "42",
        "--world",
        yaml_path,
        "--no-traffic",
    ]
    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        for td in (td_a, td_b):
            r = subprocess.run(
                [*runner_args, "--results-dir", td],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert r.returncode == 0, (
                f"TC15 runner exit {r.returncode}; stderr={r.stderr[-400:]}"
            )

        jsonl_a = Path(td_a) / "arena_v1" / "a_star_once" / "42.trace.jsonl"
        jsonl_b = Path(td_b) / "arena_v1" / "a_star_once" / "42.trace.jsonl"
        assert jsonl_a.exists() and jsonl_b.exists(), (
            f"TC15 trace JSONLs missing: a_exists={jsonl_a.exists()}, "
            f"b_exists={jsonl_b.exists()}"
        )
        assert filecmp.cmp(str(jsonl_a), str(jsonl_b), shallow=False), (
            "TC15 trace JSONLs differ — same-seed determinism broken (issue lives in "
            "runners/run_episode.py, not arena.py)"
        )

        json_a = json.loads(
            (Path(td_a) / "arena_v1" / "a_star_once" / "42.json").read_text(encoding="utf-8")
        )
        json_b = json.loads(
            (Path(td_b) / "arena_v1" / "a_star_once" / "42.json").read_text(encoding="utf-8")
        )
        json_a.pop("wallclock_per_step", None)
        json_b.pop("wallclock_per_step", None)
        assert json_a == json_b, (
            f"TC15 metrics differ (excluding wallclock_per_step): "
            f"a={json_a} b={json_b}"
        )


def tc16(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal world
    """Planner failure path: arena_no_path.yaml yields planner_error and no trace JSONL."""
    repo_root = Path(__file__).resolve().parent.parent
    no_path_yaml = str(repo_root / "arena" / "arena_no_path.yaml")
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "runners.run_episode",
                "--algorithm",
                "a_star_once",
                "--seed",
                "0",
                "--world",
                no_path_yaml,
                "--results-dir",
                td,
                "--no-traffic",
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode == 0, (
            f"TC16 runner exit {r.returncode}; stderr={r.stderr[-400:]}"
        )

        json_path = Path(td) / "arena_no_path" / "a_star_once" / "0.json"
        jsonl_path = Path(td) / "arena_no_path" / "a_star_once" / "0.trace.jsonl"
        assert json_path.exists(), f"TC16 metrics JSON missing at {json_path}"
        assert not jsonl_path.exists(), (
            f"TC16 trace JSONL must NOT exist on planner failure; found {jsonl_path}"
        )

        metrics = json.loads(json_path.read_text(encoding="utf-8"))
        assert metrics["planner_error"] is not None, (
            f"TC16 planner_error must not be None: {metrics}"
        )
        assert "could not find a path" in metrics["planner_error"], (
            f"TC16 planner_error must contain 'could not find a path': {metrics}"
        )
        assert metrics["time_to_goal"] is None, (
            f"TC16 time_to_goal must be None on planner failure: {metrics}"
        )
        assert metrics["crashed"] is False, f"TC16 crashed flag: {metrics}"
        assert metrics["timed_out"] is False, f"TC16 timed_out flag: {metrics}"


# ---------------------------------------------------------------------------
# TC17..TC23 — Phase 2 traffic checks (TC17..TC21) + path partitioning (TC22)
# + import-cycle guard (TC23). All use arena/arena_v1.yaml unless noted.
# ---------------------------------------------------------------------------


def tc17(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed (0) for determinism
    """Init population: 20 obstacles on perimeter edges with inward headings."""
    from arena.dynamic import DynamicObstacleState, TARGET_POPULATION, OBSTACLE_RADIUS

    arena = Arena(yaml_path, seed=0, traffic=True)
    try:
        _, _, info = arena.reset()
        assert info.dynamic_obstacle_count == TARGET_POPULATION, (
            f"TC17: dynamic_obstacle_count must be {TARGET_POPULATION}, got {info.dynamic_obstacle_count}"
        )
        snapshot = arena.initial_dynamic_snapshot
        assert len(snapshot) == TARGET_POPULATION, (
            f"TC17: snapshot length must be {TARGET_POPULATION}, got {len(snapshot)}"
        )
        # Per-obstacle invariants
        # Read the world dims from the YAML so the perimeter check tracks any future arena size change.
        import yaml as _yaml
        with open(yaml_path, "r", encoding="utf-8") as fh:
            world_data = _yaml.safe_load(fh)
        W = float(world_data["world"]["width"])
        H = float(world_data["world"]["height"])

        tol = 1e-6
        for i, obs in enumerate(snapshot):
            assert isinstance(obs, DynamicObstacleState), (
                f"TC17: snapshot[{i}] is {type(obs).__name__}, expected DynamicObstacleState"
            )
            assert obs.radius == OBSTACLE_RADIUS, (
                f"TC17: snapshot[{i}].radius must be {OBSTACLE_RADIUS}, got {obs.radius}"
            )
            # Perimeter check: must lie on one of the four edges within tol.
            on_south = abs(obs.y - 0.0) < tol
            on_north = abs(obs.y - H) < tol
            on_west  = abs(obs.x - 0.0) < tol
            on_east  = abs(obs.x - W) < tol
            assert on_south or on_north or on_west or on_east, (
                f"TC17: snapshot[{i}] at ({obs.x}, {obs.y}) is not on a perimeter edge "
                f"(W={W}, H={H}, tol={tol})"
            )
            # Inward-heading check: the velocity must have a non-negative inward
            # component for AT LEAST ONE edge the obstacle lies on. The spawner draws
            # heading from a half-open cone, so the inward component can be exactly 0 at
            # a cone endpoint (non-strict), and a corner spawn lies on two edges while
            # only the edge it was drawn from is guaranteed inward — so require ANY
            # satisfying edge rather than asserting every edge it touches.
            inward = (
                (on_south and obs.vy >= 0.0)
                or (on_north and obs.vy <= 0.0)
                or (on_west and obs.vx >= 0.0)
                or (on_east and obs.vx <= 0.0)
            )
            assert inward, (
                f"TC17: snapshot[{i}] at ({obs.x}, {obs.y}) vel ({obs.vx}, {obs.vy}) "
                f"is not inward for any edge it lies on "
                f"(S={on_south}, N={on_north}, W={on_west}, E={on_east})"
            )
            # Speed in [0.3, 1.5] m/s (factors of MAX_LINEAR_SPEED=1.0).
            speed = (obs.vx**2 + obs.vy**2) ** 0.5
            assert 0.3 - tol <= speed <= 1.5 + tol, (
                f"TC17: snapshot[{i}] speed must be in [0.3, 1.5], got {speed}"
            )
    finally:
        arena.close()


def tc18(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed
    """Refill maintains population at 20 across a full-traversal window, with at least one despawn."""
    from arena.dynamic import TARGET_POPULATION

    arena = Arena(yaml_path, seed=1, traffic=True)
    try:
        _, _, _ = arena.reset()
        initial_live_ids = set(arena.initial_dynamic_snapshot[i].id for i in range(TARGET_POPULATION))

        # Run enough ticks for the slowest obstacle (0.3 m/s) to traverse 50 m at dt=0.1:
        # 50 / 0.3 ≈ 167 ticks, plus 50 margin.
        max_ticks = int(50.0 / 0.3 / arena._dt) + 50
        zero = np.array([[0.0], [0.0]], dtype=float)
        for _ in range(max_ticks):
            _, _, _, info = arena.step(zero)
            assert info.dynamic_obstacle_count == TARGET_POPULATION, (
                f"TC18: dynamic_obstacle_count fell to {info.dynamic_obstacle_count} at step {info.step_idx}; "
                f"refill broken"
            )
            if info.crashed or info.timed_out or info.reached_goal:
                # Done early — should not happen with a stationary robot in arena_v1's
                # safe (2,2) start, but break cleanly if it does.
                break
        # initial_dynamic_snapshot is frozen at t=0, so read the final live set
        # straight from the spawner to detect despawn churn.
        assert arena._spawner is not None
        final_live_ids = set(arena._spawner.live_ids)
        churned = initial_live_ids.symmetric_difference(final_live_ids)
        assert len(churned) > 0, (
            f"TC18: expected at least one despawn over {max_ticks} ticks, but the live-id set "
            f"is unchanged ({len(initial_live_ids)} ids). Despawn path may be broken."
        )
    finally:
        arena.close()


def tc19(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed
    """Robot-vs-dynamic-obstacle collision fires info.crashed via _inject_for_test."""
    arena = Arena(yaml_path, seed=2, traffic=True)
    try:
        _, _, _ = arena.reset()
        assert arena._spawner is not None, "TC19: spawner must be live with traffic=True"
        # Inject an obstacle 1.0 m east of (2,2), moving west at 1.0 m/s.
        # Collision contact distance = robot_radius (0.2) + obstacle_radius (0.3) = 0.5 m.
        # Obstacle reaches contact distance after moving 0.5 m → 5 ticks at dt=0.1.
        arena._spawner._inject_for_test(x=3.0, y=2.0, vx=-1.0, vy=0.0)
        zero = np.array([[0.0], [0.0]], dtype=float)
        crashed = False
        for _ in range(20):
            _, _, _, info = arena.step(zero)
            if info.crashed:
                crashed = True
                break
        assert crashed, (
            "TC19: robot did not crash within 20 ticks of an obstacle traveling toward it at 1 m/s "
            "from 1 m east — irsim collision detection on dynamic obstacles may be broken"
        )
    finally:
        arena.close()


def tc20(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed
    """Traffic determinism: two same-seed arenas produce identical dynamic_obstacles_sha256 sequences."""
    seed_value = 3
    n_ticks = 200
    zero = np.array([[0.0], [0.0]], dtype=float)

    def collect_hashes() -> list[str]:
        arena = Arena(yaml_path, seed=seed_value, traffic=True)
        try:
            _, _, info0 = arena.reset()
            hashes: list[str] = []
            assert info0.dynamic_obstacles_sha256 is not None, (
                "TC20: reset() must produce a non-None dynamic_obstacles_sha256 when traffic=True"
            )
            hashes.append(info0.dynamic_obstacles_sha256)
            for _ in range(n_ticks):
                _, _, _, info = arena.step(zero)
                assert info.dynamic_obstacles_sha256 is not None, (
                    f"TC20: step {info.step_idx} sha256 is None with traffic=True"
                )
                hashes.append(info.dynamic_obstacles_sha256)
                if info.crashed or info.timed_out or info.reached_goal:
                    break
            return hashes
        finally:
            arena.close()

    hashes_a = collect_hashes()
    hashes_b = collect_hashes()
    assert hashes_a == hashes_b, (
        f"TC20: dynamic_obstacles_sha256 sequences differ between two same-seed runs. "
        f"len_a={len(hashes_a)}, len_b={len(hashes_b)}. First mismatch at tick "
        f"{next((i for i, (a, b) in enumerate(zip(hashes_a, hashes_b)) if a != b), 'n/a')}"
    )


def tc21(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed
    """Snapshot shape, type, and immutability."""
    import dataclasses as _dc
    from arena.dynamic import DynamicObstacleState, OBSTACLE_RADIUS, TARGET_POPULATION

    # traffic=False, pre-reset: ()
    arena_off = Arena(yaml_path, seed=5, traffic=False)
    try:
        assert arena_off.initial_dynamic_snapshot == (), (
            f"TC21: traffic=False, pre-reset snapshot must be (), got {arena_off.initial_dynamic_snapshot}"
        )
        # traffic=False, post-reset: still ()
        arena_off.reset()
        assert arena_off.initial_dynamic_snapshot == (), (
            f"TC21: traffic=False, post-reset snapshot must be (), got {arena_off.initial_dynamic_snapshot}"
        )
    finally:
        arena_off.close()

    # traffic=True
    arena_on = Arena(yaml_path, seed=5, traffic=True)
    try:
        # Pre-reset: ()
        assert arena_on.initial_dynamic_snapshot == (), (
            f"TC21: traffic=True, pre-reset snapshot must be (), got len={len(arena_on.initial_dynamic_snapshot)}"
        )
        # Post-reset: 20 frozen entries
        arena_on.reset()
        snap = arena_on.initial_dynamic_snapshot
        assert isinstance(snap, tuple), f"TC21: snapshot must be tuple, got {type(snap).__name__}"
        assert len(snap) == TARGET_POPULATION, f"TC21: snapshot len must be {TARGET_POPULATION}, got {len(snap)}"
        first = snap[0]
        assert _dc.is_dataclass(first), f"TC21: snapshot[0] must be a dataclass, got {type(first).__name__}"
        assert first.radius == OBSTACLE_RADIUS, (
            f"TC21: snapshot[0].radius must be {OBSTACLE_RADIUS}, got {first.radius}"
        )
        # Frozen: attempting to mutate must raise FrozenInstanceError
        try:
            first.x = 999.0  # type: ignore[misc]
        except _dc.FrozenInstanceError:
            pass
        else:
            raise AssertionError("TC21: DynamicObstacleState must be frozen; field assignment did not raise")
    finally:
        arena_on.close()


def tc22(yaml_path: str, seed: int) -> None:  # noqa: ARG001
    """World-stem partitioning: same seed, two different worlds, two distinct result files."""
    repo_root = Path(__file__).resolve().parent.parent
    v1_yaml = str(repo_root / "arena" / "arena_v1.yaml")
    v2_yaml = str(repo_root / "arena" / "arena_v2_hard.yaml")
    common = [
        sys.executable, "-m", "runners.run_episode",
        "--algorithm", "a_star_once",
        "--seed", "42",
        "--no-traffic",  # so A* succeeds on both worlds
    ]
    with tempfile.TemporaryDirectory() as td:
        for world in (v1_yaml, v2_yaml):
            r = subprocess.run(
                [*common, "--world", world, "--results-dir", td],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert r.returncode == 0, (
                f"TC22 runner failed on {world}: exit={r.returncode}; stderr={r.stderr[-400:]}"
            )

        json_v1 = Path(td) / "arena_v1" / "a_star_once" / "42.json"
        json_v2 = Path(td) / "arena_v2_hard" / "a_star_once" / "42.json"
        jsonl_v1 = Path(td) / "arena_v1" / "a_star_once" / "42.trace.jsonl"
        jsonl_v2 = Path(td) / "arena_v2_hard" / "a_star_once" / "42.trace.jsonl"

        for p in (json_v1, json_v2, jsonl_v1, jsonl_v2):
            assert p.exists(), f"TC22: expected output missing at {p}"

        data_v1 = json.loads(json_v1.read_text(encoding="utf-8"))
        data_v2 = json.loads(json_v2.read_text(encoding="utf-8"))
        # Different worlds at the same seed must produce different runs.
        # The simplest non-trivial check: at least one of (path_length, time_to_goal) differs.
        differs = (
            data_v1.get("path_length") != data_v2.get("path_length")
            or data_v1.get("time_to_goal") != data_v2.get("time_to_goal")
        )
        assert differs, (
            f"TC22: arena_v1 and arena_v2_hard at seed=42 produced identical metrics; "
            f"world-stem partitioning is silently clobbering. v1={data_v1}, v2={data_v2}"
        )


def tc23(yaml_path: str, seed: int) -> None:  # noqa: ARG001
    """Import-cycle guard: both import orders succeed in a clean subprocess."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    for order in ("import planners; import arena.arena",
                  "import arena.arena; import planners"):
        r = subprocess.run(
            [sys.executable, "-c", order],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert r.returncode == 0, (
            f"TC23: import order failed (`{order}`): exit={r.returncode}; "
            f"stderr={r.stderr[-400:]}"
        )


def tc24(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed (7)
    """Traffic-ON runner end-to-end: 8-key trace + byte-identical across two seeded runs.

    The runner's shipped default is traffic=True, but TC14/TC15/TC16/TC22 all force
    --no-traffic, so without this case the default code path is untested: the 8th
    trace key wiring (step-0 reset sha + per-step post-step sha) and trace-level
    determinism through the runner under traffic.
    """
    repo_root = Path(__file__).resolve().parent.parent
    world_stem = Path(yaml_path).stem
    cmd = [
        sys.executable, "-m", "runners.run_episode",
        "--algorithm", "a_star_once",
        "--seed", "7",
        "--world", yaml_path,
        "--traffic",
    ]
    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        for td in (td_a, td_b):
            r = subprocess.run(
                [*cmd, "--results-dir", td],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert r.returncode == 0, (
                f"TC24 runner exit {r.returncode}; stderr={r.stderr[-400:]}"
            )

        jsonl_a = Path(td_a) / world_stem / "a_star_once" / "7.trace.jsonl"
        jsonl_b = Path(td_b) / world_stem / "a_star_once" / "7.trace.jsonl"
        assert jsonl_a.exists() and jsonl_b.exists(), (
            f"TC24 trace JSONLs missing: a={jsonl_a.exists()}, b={jsonl_b.exists()}"
        )

        lines_a = jsonl_a.read_text(encoding="utf-8").splitlines()
        assert lines_a, "TC24: traffic trace JSONL is empty"
        for ln, raw in enumerate(lines_a):
            rec = json.loads(raw)
            assert "dynamic_obstacles_sha256" in rec, (
                f"TC24: trace line {ln} missing dynamic_obstacles_sha256 with traffic on; "
                f"keys={sorted(rec)}"
            )
            assert len(rec) == 8, (
                f"TC24: trace line {ln} must have 8 keys with traffic on, got {len(rec)}: {sorted(rec)}"
            )

        assert filecmp.cmp(jsonl_a, jsonl_b, shallow=False), (
            "TC24: two same-seed traffic runs produced differing trace JSONL; "
            "traffic determinism through the runner is broken"
        )


def tc25(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure computation, no world used
    """Phase 3 seed derivation: determinism, uniqueness, prefix property, master-sensitivity."""
    from runners.run_experiment import derive_episode_seeds

    fifty = derive_episode_seeds(7, 50)
    assert len(fifty) == 50, f"TC25: expected 50 seeds, got {len(fifty)}"
    assert len(set(fifty)) == 50, "TC25: derived seeds are not unique"
    assert all(isinstance(s, int) and s >= 0 for s in fifty), (
        "TC25: seeds must be non-negative ints"
    )
    assert derive_episode_seeds(7, 50) == fifty, "TC25: derivation is not deterministic"
    assert derive_episode_seeds(7, 3) == fifty[:3], (
        "TC25: prefix property broken (spawn(3) != spawn(50)[:3])"
    )
    assert derive_episode_seeds(8, 3) != fifty[:3], (
        "TC25: a different master seed produced an identical prefix"
    )


def tc26(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses arena_no_path fixture
    """Phase 3 batch determinism + parallel-ordering.

    Runs the batch runner on the boxed-in-start world (A* fails fast, so each episode
    terminates in seconds with no driving loop). A and B at --jobs 1 must be byte-identical;
    C at --jobs 3 must keep the manifest in derivation order (completion order must not leak).
    """
    repo_root = Path(__file__).resolve().parent.parent
    world = str(repo_root / "arena" / "arena_no_path.yaml")
    # Master seed 1 yields a DESCENDING 3-seed prefix, so derivation order differs from
    # sort-by-seed order — this is what gives the ordering assertion below real teeth.
    base = [
        sys.executable, "-m", "runners.run_experiment",
        "--algorithm", "a_star_once",
        "--world", world,
        "--master-seed", "1",
        "--num-seeds", "3",
        "--no-traffic",
    ]

    def _run(td: str, extra: list[str]) -> Path:
        r = subprocess.run(
            [*base, "--results-dir", td, *extra],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, (
            f"TC26 batch failed (extra={extra}): exit={r.returncode}; stderr={r.stderr[-400:]}"
        )
        return Path(td) / "arena_no_path" / "a_star_once"

    def _manifest_no_git(out_dir: Path) -> dict:
        m = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))
        m.pop("git_sha", None)  # robust to dirty tree / absent git
        return m

    # ignore_cleanup_errors: child subprocesses wrote into these dirs; on Windows a lingering
    # handle or an AV/indexer lock can make rmtree raise PermissionError at block exit, which
    # would fail --check for a reason unrelated to the assertions.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_a, \
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_b, \
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_c:
        dir_a = _run(td_a, ["--jobs", "1"])
        dir_b = _run(td_b, ["--jobs", "1"])
        dir_c = _run(td_c, ["--jobs", "3"])

        seeds = sorted(int(p.stem) for p in dir_a.glob("[0-9]*.json"))
        assert len(seeds) == 3, f"TC26: expected 3 episode JSONs, got {len(seeds)}"

        for s in seeds:
            assert filecmp.cmp(dir_a / f"{s}.json", dir_b / f"{s}.json", shallow=False), (
                f"TC26: per-seed metrics JSON differ across two same-master-seed runs (seed={s})"
            )
            assert not (dir_a / f"{s}.trace.jsonl").exists(), (
                f"TC26: planner-failure world wrote a trace for seed={s}"
            )

        man_a = _manifest_no_git(dir_a)
        assert man_a == _manifest_no_git(dir_b), (
            "TC26: manifests differ across two same-master-seed --jobs 1 runs"
        )

        # Assert the manifest order against the KNOWN derivation order, not against itself.
        # With a descending prefix this catches a sorted-by-seed build AND a completion-order
        # leak in the --jobs 3 path; the old order_a == order_c check compared two outputs of the
        # same code against ascending default-master seeds and so could distinguish neither.
        from runners.run_experiment import derive_episode_seeds

        derived = list(derive_episode_seeds(1, 3))
        assert derived != sorted(derived), "TC26: chosen master must give a non-monotonic prefix"
        man_c = _manifest_no_git(dir_c)
        order_a = [e["seed"] for e in man_a["episodes"]]
        order_c = [e["seed"] for e in man_c["episodes"]]
        assert order_a == derived, (
            f"TC26: --jobs 1 manifest not in derivation order: {order_a} != {derived}"
        )
        assert order_c == derived, (
            f"TC26: --jobs 3 reordered the manifest episodes (completion order leaked in): "
            f"{order_c} != {derived}"
        )
        assert man_a["derived_seeds"] == derived, (
            "TC26: manifest derived_seeds not in derivation order"
        )
        assert man_a["derived_seeds"] == man_c["derived_seeds"], (
            "TC26: derived_seeds differ between --jobs 1 and --jobs 3"
        )


def tc27(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — writes its own malformed world
    """Phase 3 failure accounting: a malformed (but existing) world makes every child exit
    non-zero; the batch continues, reports the failures, and itself exits non-zero."""
    repo_root = Path(__file__).resolve().parent.parent
    # ignore_cleanup_errors: a child subprocess wrote into this dir; on Windows a lingering
    # handle / AV / indexer lock can make rmtree raise PermissionError at block exit.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        bad_yaml = Path(td) / "bad.yaml"
        bad_yaml.write_text("not: [valid: arena", encoding="utf-8")  # irsim/yaml rejects this
        r = subprocess.run(
            [
                sys.executable, "-m", "runners.run_experiment",
                "--algorithm", "a_star_once",
                "--world", str(bad_yaml),
                "--num-seeds", "2",
                "--no-traffic",
                "--results-dir", td,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode != 0, (
            f"TC27: batch should exit non-zero when all seeds fail; got {r.returncode}"
        )
        # The manifest checks below are the authoritative assertions: they prove both children
        # were recorded as runner failures with non-zero exit codes. They also guard the fixture
        # itself — if this "malformed" world ever parsed and failed only at the planner stage, the
        # child would exit 0, flipping those checks to status "ok"/exit_code 0 (a loud failure)
        # instead of a silent pass. We keep one light console check (the failure-detail section
        # prints) but deliberately do NOT assert the exact "<n> runner-failed" wording, which would
        # couple the test to console phrasing for no added coverage.
        assert "runner failures:" in r.stdout, (
            "TC27: summary omitted the per-seed failure detail section"
        )

        manifest = json.loads(
            (Path(td) / "bad" / "a_star_once" / "_manifest.json").read_text(encoding="utf-8")
        )
        statuses = [e["status"] for e in manifest["episodes"]]
        assert statuses == ["runner_error", "runner_error"], (
            f"TC27: manifest episodes should both be runner_error, got {statuses}"
        )
        assert all(e["exit_code"] != 0 for e in manifest["episodes"]), (
            "TC27: failed episodes must record a non-zero exit_code"
        )


# ---------------------------------------------------------------------------
# CLI runner — --check (default) or --render. See module docstring above.
# ---------------------------------------------------------------------------


def _run_checks(yaml_path: str, seed: int) -> int:
    cases: list[tuple[str, Any]] = [
        ("TC1: construct + close", tc1),
        ("TC2: reset shapes & info", tc2),
        ("TC2b: missing-lidar tick", tc2b),
        ("TC3: one step", tc3),
        ("TC4: deliberate crash within 200 steps", tc4),
        ("TC5: timeout fires", tc5),
        ("TC6: step after done raises", tc6),
        ("TC7: reset after done clears state", tc7),
        ("TC8: arrive_flag injection sets reached_goal", tc8),
        ("TC9: action validation", tc9),
        ("TC10: manual_astar inflation check", tc10),
        ("TC11: YAML schema fields", tc11),
        ("TC12: lidar beam mismatch raises ArenaConfigError", tc12),
        ("TC13: wall crash via teleport", tc13),
        ("TC14: full A* drive via runner", tc14),
        ("TC15: determinism — same seed -> byte-identical trace", tc15),
        ("TC16: planner failure on arena_no_path.yaml", tc16),
        ("TC17: init population (20 on edges, inward)", tc17),
        ("TC18: refill maintained across full-traversal window", tc18),
        ("TC19: robot-vs-dynamic-obstacle collision via _inject_for_test", tc19),
        ("TC20: traffic determinism — sha256 sequences match", tc20),
        ("TC21: snapshot shape, type, immutability", tc21),
        ("TC22: world-stem partitioning end-to-end", tc22),
        ("TC23: import-cycle guard (planners <-> arena.arena)", tc23),
        ("TC24: traffic-ON runner — 8-key trace + determinism", tc24),
        ("TC25: Phase 3 seed derivation (determinism/uniqueness/prefix)", tc25),
        ("TC26: Phase 3 batch determinism + parallel-ordering", tc26),
        ("TC27: Phase 3 failure accounting + non-zero batch exit", tc27),
    ]
    failures = 0
    for label, fn in cases:
        try:
            fn(yaml_path, seed)
            print(f"PASS - {label}")
        except Exception as exc:
            print(f"FAIL - {label}: {type(exc).__name__}: {exc}")
            failures += 1
    return failures


def _run_render(yaml_path: str, seed: int) -> None:
    arena = Arena(yaml_path, seed, render=True)
    try:
        arena.reset()
        zero = np.array([[0.0], [0.0]], dtype=float)
        while True:
            _, _, done, info = arena.step(zero)
            if done:
                print(f"done: {info}")
                break
    finally:
        arena.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arena smoke/check harness")
    parser.add_argument(
        "yaml_path",
        help="Path to arena world YAML (e.g. arena/arena_v1.yaml)",
    )
    parser.add_argument("--seed", type=int, default=42)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--render",
        action="store_true",
        help="Interactive smoke loop (visible window)",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Run TC1-TC27 headless (28 cases, incl. Phase 2 traffic + Phase 3 batch runner)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import sys

    args = _parse_args()
    if args.render:
        _run_render(args.yaml_path, args.seed)
    else:
        # Default to --check when neither flag given.
        sys.exit(_run_checks(args.yaml_path, args.seed))
