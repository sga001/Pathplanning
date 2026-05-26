"""Single-episode runner — drives a registered planner against an Arena world and writes metrics + trace.

CLI:
    python -m runners.run_episode \
        --algorithm <name>      # required; e.g. "a_star_once"
        --seed <int>            # required
        --world <yaml_path>     # required; e.g. arena/arena_v1.yaml
        [--render]              # optional flag; default False
        [--results-dir <dir>]   # optional; default "results"

Programmatic:
    from runners.run_episode import main
    rc = main(["--algorithm", "a_star_once", "--seed", 42,
               "--world", "arena/arena_v1.yaml", "--results-dir", "out"])

Outputs:
    <results-dir>/<algorithm>/<seed>.json         — 7-field metrics JSON (always written)
    <results-dir>/<algorithm>/<seed>.trace.jsonl  — per-step trace (only on planning success)

Exit codes:
    0 — episode terminated (success, crash, timeout, or planner failure all return 0)
    2 — argparse parsing error or Arena __init__ config error (e.g., ArenaConfigError)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np

# Make repo root importable so `from manual_astar import ...` works when this
# module is invoked as `python -m runners.run_episode` from any cwd.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from arena.arena import Arena  # noqa: E402
from manual_astar import (  # noqa: E402
    WAYPOINT_REACHED_DISTANCE,
    WaypointFollower,
    compute_action,
)
from planners import AStarOncePlanner, PathPlanner  # noqa: E402


# Algorithm registry. Phase 6 expands this dict with the remaining planners.
ALGORITHMS: dict[str, type[PathPlanner]] = {
    "a_star_once": AStarOncePlanner,
}

# Metrics JSON has exactly these seven keys. Extends Mission.md Phase 1's
# six-field list with `planner_error` (str | None).
METRICS_KEYS = (
    "time_to_goal",
    "crashed",
    "timed_out",
    "path_length",
    "mean_speed",
    "wallclock_per_step",
    "planner_error",
)


@dataclass(frozen=True)
class RunnerArgs:
    """Parsed CLI arguments — frozen so accidental mutation is impossible."""

    algorithm: str
    seed: int
    world: str
    render: bool
    results_dir: str


def _parse_args(argv: list[str] | None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        prog="runners.run_episode",
        description="Run a single planner episode against an Arena world.",
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=list(ALGORITHMS),
        help="Registered planner name (e.g. 'a_star_once').",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Seed forwarded to Arena (controls traffic + motion RNGs).",
    )
    parser.add_argument(
        "--world",
        required=True,
        help="Path to the world YAML (e.g. arena/arena_v1.yaml).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open an irsim render window (default: headless).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Output directory root; results go in <results-dir>/<algorithm>/.",
    )
    ns = parser.parse_args(argv)
    return RunnerArgs(
        algorithm=ns.algorithm,
        seed=int(ns.seed),
        world=ns.world,
        render=bool(ns.render),
        results_dir=ns.results_dir,
    )


def _trace_line(
    *,
    step: int,
    state: Any,
    action: Any,
    lidar: np.ndarray,
    crashed: bool,
    reached_goal: bool,
    done: bool,
) -> str:
    """Serialize one trace record to a compact, order-stable JSON string.

    `sort_keys=True` and `separators=(",", ":")` are mandatory — they guarantee
    byte-identical lines across two same-seed runs (TC15's identity check).
    """
    record = {
        "step": int(step),
        "state": [float(x) for x in state],
        "action": [float(x) for x in action],
        "lidar_sha256": hashlib.sha256(lidar.tobytes()).hexdigest(),
        "crashed": bool(crashed),
        "reached_goal": bool(reached_goal),
        "done": bool(done),
    }
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _write_trace_line(file: IO[str], **kwargs: Any) -> None:
    file.write(_trace_line(**kwargs))
    file.write("\n")


def _write_metrics(
    metrics_path: Path,
    *,
    time_to_goal: float | None,
    crashed: bool,
    timed_out: bool,
    path_length: float,
    mean_speed: float,
    wallclock_per_step: float,
    planner_error: str | None,
) -> None:
    """Write the 7-key metrics JSON. Always exactly these keys, in any order."""
    metrics = {
        "time_to_goal": None if time_to_goal is None else float(time_to_goal),
        "crashed": bool(crashed),
        "timed_out": bool(timed_out),
        "path_length": float(path_length),
        "mean_speed": float(mean_speed),
        "wallclock_per_step": float(wallclock_per_step),
        "planner_error": None if planner_error is None else str(planner_error),
    }
    # Defensive: guarantee the schema before disk write.
    if set(metrics) != set(METRICS_KEYS):
        raise RuntimeError(
            f"metrics keys mismatch: got {sorted(metrics)}, expected {sorted(METRICS_KEYS)}"
        )
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, sort_keys=True, indent=2)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Run one episode end-to-end. See module docstring for CLI semantics."""
    args = _parse_args(argv)

    planner_cls = ALGORITHMS[args.algorithm]
    planner = planner_cls()

    # Arena __init__ may raise ArenaConfigError — let it propagate (exit 2 via
    # the harness convention; the OS surfaces an unhandled exception as nonzero
    # but argparse-style exit 2 happens above before this line).
    arena = Arena(args.world, args.seed, render=args.render)

    out_dir = Path(args.results_dir) / args.algorithm
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{args.seed}.json"
    trace_path = out_dir / f"{args.seed}.trace.jsonl"

    trace_file: IO[str] | None = None

    try:
        state0, lidar0, _info0 = arena.reset()

        # Open the trace file and emit the step-0 anchor BEFORE planning so the
        # post-reset state is always captured even if planning fails mid-way —
        # the planner-failure branch then deletes the trace (no trace on failure).
        trace_file = open(trace_path, "w", encoding="utf-8")
        _write_trace_line(
            trace_file,
            step=0,
            state=state0.tolist(),
            action=[0.0, 0.0],
            lidar=lidar0,
            crashed=False,
            reached_goal=False,
            done=False,
        )

        # Plan. Only (ValueError, RuntimeError) are caught — these are the
        # exception classes manual_astar.{validate_start_and_goal,astar_search}
        # produce. Any other exception (TypeError, AttributeError, ImportError,
        # ...) is a programmer bug and must surface loudly.
        try:
            waypoints = planner.plan(
                args.world, arena.initial_dynamic_snapshot, lidar0
            )
        except (ValueError, RuntimeError) as exc:
            # Planner failure: tear down the trace (no trace on planner failure)
            # and write metrics with planner_error set.
            trace_file.close()
            trace_file = None
            try:
                trace_path.unlink()
            except FileNotFoundError:
                pass
            _write_metrics(
                metrics_path,
                time_to_goal=None,
                crashed=False,
                timed_out=False,
                path_length=0.0,
                mean_speed=0.0,
                wallclock_per_step=0.0,
                planner_error=str(exc),
            )
            return 0

        if not waypoints:
            # `plan()` returned an empty tuple — treat as planner failure
            # (no path to follow). Same disposition as the exception branch.
            trace_file.close()
            trace_file = None
            try:
                trace_path.unlink()
            except FileNotFoundError:
                pass
            _write_metrics(
                metrics_path,
                time_to_goal=None,
                crashed=False,
                timed_out=False,
                path_length=0.0,
                mean_speed=0.0,
                wallclock_per_step=0.0,
                planner_error="planner returned an empty waypoint list",
            )
            return 0

        # Drive the planned waypoints until Arena reports done.
        follower = WaypointFollower(list(waypoints), WAYPOINT_REACHED_DISTANCE)
        path_length = 0.0
        total_wallclock = 0.0
        prev_xy = state0[:2].copy()
        step_count = 0
        done = False
        info = None

        while not done:
            action = compute_action(arena._robot, follower)
            state, lidar, done, info = arena.step(action)
            step_count += 1
            path_length += float(np.linalg.norm(state[:2] - prev_xy))
            prev_xy = state[:2].copy()
            total_wallclock += info.wallclock_per_step
            _write_trace_line(
                trace_file,
                step=info.step_idx,
                state=state.tolist(),
                action=action.flatten().tolist(),
                lidar=lidar,
                crashed=info.crashed,
                reached_goal=info.reached_goal,
                done=done,
            )

        # Defensive: the only way `info` is None here is a zero-iteration loop,
        # which can't happen because `done` starts False. Keep the guard so a
        # future refactor that changes the loop contract surfaces loudly.
        if info is None:
            raise RuntimeError("episode loop terminated without producing an EpisodeInfo")

        # Flush + fsync the trace before close so TC15's two-subprocess byte
        # comparison cannot race on dirty page cache.
        trace_file.flush()
        os.fsync(trace_file.fileno())
        trace_file.close()
        trace_file = None

        sim_time = float(info.sim_time)
        time_to_goal = sim_time if info.reached_goal else None
        mean_speed = (path_length / sim_time) if sim_time > 0.0 else 0.0
        wallclock_per_step = (
            total_wallclock / step_count if step_count > 0 else 0.0
        )

        _write_metrics(
            metrics_path,
            time_to_goal=time_to_goal,
            crashed=bool(info.crashed),
            timed_out=bool(info.timed_out),
            path_length=path_length,
            mean_speed=mean_speed,
            wallclock_per_step=wallclock_per_step,
            planner_error=None,
        )
        return 0
    finally:
        # Best-effort cleanup. We never want a leaked file handle or irsim env
        # to bleed into another process — but we also must not mask the
        # primary exception with a cleanup error.
        if trace_file is not None:
            try:
                trace_file.close()
            except Exception:
                pass
        try:
            arena.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
