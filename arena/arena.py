from __future__ import annotations

import argparse
import dataclasses
import os
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


class ArenaConfigError(ValueError):
    """Raised at Arena.__init__ for malformed config (e.g. lidar beam count mismatch)."""


class ArenaRuntimeError(RuntimeError):
    """Raised mid-episode for irsim contract violations (e.g. lidar dict missing 'ranges')."""


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


class Arena:
    """Static 50x50 arena wrapping irsim. Phase 0 = no dynamic obstacles."""

    def __init__(
        self,
        yaml_path: str | Path,
        seed: int,
        render: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._yaml_path = Path(yaml_path)
        self._render = bool(render)
        self._timeout_s = float(timeout_s)
        self._master_seed = int(seed)

        self._env = irsim.make(str(self._yaml_path), display=self._render)
        self._robot = self._env.robot_list[0]
        self._dt = float(self._env.step_time)
        self._goal_xy = self._robot.goal[:2, 0].astype(np.float64)

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

        self._step_idx = 0
        self._done = False
        self._closed = False

    def reset(self) -> tuple[np.ndarray, np.ndarray, EpisodeInfo]:
        if self._closed:
            raise RuntimeError("Arena is closed")

        self._env.reset()
        # Defensive re-clear: irsim's reset() runs an internal warm-up step that
        # re-evaluates arrive/collision flags against the just-reset pose.
        self._robot.arrive_flag = False
        self._robot.collision_flag = False

        # traffic first, motion second — Phase 2 spawner consumes in this order
        ss = np.random.SeedSequence(self._master_seed)
        traffic_seed, motion_seed = ss.spawn(2)
        self._traffic_rng = np.random.default_rng(traffic_seed)
        self._motion_rng = np.random.default_rng(motion_seed)

        self._step_idx = 0
        self._done = False

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
            dynamic_obstacle_count=0,
            lidar_status=lidar_status,
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

        # Snapshot flags BEFORE step: irsim's check_*_status overwrite them per tick
        # (see object_base.py:531-532), so harness-injected flags would be lost otherwise.
        pre_crashed = bool(getattr(self._robot, "collision_flag", False))
        pre_reached = bool(getattr(self._robot, "arrive_flag", False))

        start = time.perf_counter()
        self._env.step([action])
        wallclock = time.perf_counter() - start

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
            dynamic_obstacle_count=0,
            lidar_status=lidar_status,
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
    def initial_dynamic_snapshot(self) -> tuple[Any, ...]:
        """Snapshot of dynamic obstacles at t=0. Empty in Phase 0; Phase 2 narrows the type."""
        return ()

    def close(self) -> None:
        if self._closed:
            return
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

        new_state = np.array([[20.0], [19.0], [np.pi / 2]], dtype=float)
        try:
            arena._robot.state = new_state
        except (AttributeError, TypeError):
            arena._robot._state = new_state

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
        help="Run TC1-TC12 headless",
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
