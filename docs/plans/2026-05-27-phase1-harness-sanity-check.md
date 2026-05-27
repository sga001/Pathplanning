# Phase 1 — Harness Sanity Check Plan

**Goal:** Prove the Arena harness can host a real planner end-to-end by running A* on the static `arena_v1.yaml` world, verifying irsim's collision/arrive flags fire on scripted drives, writing per-episode metrics + a deterministic trace to `results/<algorithm>/<seed>.{json,trace.jsonl}`, and demonstrating same-seed → byte-identical traces. This phase gates all subsequent algorithm work in Mission.md.

**Approach:** Land two new top-level packages — `planners/` and `runners/` — that together implement the Phase 6 planner-adapter pattern in miniature, but with only A* shipped. `planners/a_star.py` exposes an `AStarOncePlanner` whose `plan()` method wraps `manual_astar.plan_waypoints()` behind a stable `PathPlanner` Protocol. `runners/run_episode.py` is the argparse CLI that constructs an `Arena`, calls the planner once at `t = 0`, runs `WaypointFollower` against `Arena.step()`, accumulates the six headline metrics + a per-step trace JSONL, and writes both to `results/<algorithm>/<seed>.*`. Three new test cases TC13–TC15 (wall crash, goal arrival via the runner, byte-identical re-run) plus TC16 (planner failure on an unsolvable Arena world) extend `arena/arena.py --check` from 13 PASS to 17 PASS. The planner-failure fixture is a new `arena/arena_no_path.yaml` (NOT the legacy `tests/no_path.yaml`, which predates the Arena lidar contract). `results/` is gitignored with a committed `.gitkeep`.

## Scope

- **In scope:**
  - `planners/__init__.py` + `planners/_types.py` — `Path` alias and `PathPlanner` Protocol.
  - `planners/a_star.py` — `AStarOncePlanner` adapter for `manual_astar.plan_waypoints`.
  - `runners/__init__.py` (empty, intentional) + `runners/run_episode.py` — argparse CLI + reusable `main(argv)` entrypoint.
  - `arena/arena.py` — add TC13 (wall crash), TC14 (full A* drive succeeds + trace content validation), TC15 (determinism), TC16 (planner-failure path on the new fixture). Register all four in `_run_checks`.
  - `arena/arena_no_path.yaml` — Phase-1-only fixture: copy of `arena_v1.yaml`'s `world`, `robot` (start, goal, lidar) blocks verbatim, but with the `obstacle[]` list replaced by a ring of rectangles fully enclosing the start `(2, 2)` so A*'s `astar_search` raises `RuntimeError("A* could not find a path from the start to the goal.")`.
  - `.gitignore` rule for `results/` + `results/.gitkeep`.
  - `CLAUDE.md` — add a "Phase 1 — Episode runner" section documenting the runner command, the `results/` layout, and TC13–TC16.
- **Out of scope:**
  - Other planners (Dijkstra, RRT, RRT*, D* Lite, DWA, APF) — Phase 6.
  - The `ReactivePlanner` Protocol (`plan() -> (v, ω)`) — Phase 6 introduces it once 2+ reactive planners triangulate the shape. Phase 1's `PathPlanner` Protocol carves only the path-planning half of Mission.md Phase 6's "dispatch on output type" rule.
  - K-sweep / replan-every-K plumbing — Phase 6b.
  - `results/plot.py` and the scatter plot — Phase 5.
  - Dynamic obstacles / `DynamicObstacle` / `TrafficSpawner` — Phase 2.
  - Refactoring `manual_astar.py` (including lifting `WorldModel` into the `planners` package). We do not delete `run_simulation` and we do not change any function's signature. Phase 6 owns the planner-interface refactor when 2+ planners exist to triangulate the shared world view.
  - Reusing the legacy `tests/no_path.yaml` — that fixture lacks the lidar block the Arena requires and stays a `manual_astar.py`-only input.
  - pytest infrastructure — the project continues the `TCi`-pattern verification inside `arena.py --check`.
  - Performance tuning (irsim render thread, batch parallelization) — outside Phase 1.

## Decisions

- **Adapter shape: `planners/a_star.py` + `runners/run_episode.py`** — Forward-compatible with Mission.md Phase 6 (one file per planner under `planners/`, single shared runner under `runners/`). User picked this option in the Phase 2 architecture batch over: (a) `arena/run_a_star.py` (one-off, dies on Phase 6 expansion), (b) `manual_astar.py --arena` mode (couples planner code to harness code; Phase 6 untangles it). The chosen option costs two extra files now and zero rework later.
- **Algorithm name `a_star_once`** — Matches Mission.md Phase 6's exact name. Using it now avoids renaming `results/a_star_once/` later when `a_star_replan_K` arrives.
- **Results path `results/<algorithm>/<seed>.json`** — Verbatim from Mission.md Phase 1. Subdirectory-per-algorithm is the natural grouping for Phase 5's plot.py. Rejected alternatives: `results/<algorithm>/seed_<N>.json` (no win), `results/<algorithm>_<seed>.json` flat (worse at 7 planners × 50 seeds = 350 files).
- **Per-step trace as JSONL with hashed lidar** — `step`, `state`, `action`, `lidar_sha256`, `done`, `crashed`, `reached_goal` per line. Hashing lidar (vs. dumping the 360-beam float array) keeps the trace small (~1 KB/step) while still catching any drift. User explicitly chose this over (a) full-lidar JSONL (~5 MB/episode × 50 seeds × 7 planners = ~1.75 GB) and (b) SHA256-of-trace-only stored in metrics JSON (cheapest, no per-step granularity). If a determinism failure surfaces, the engineer re-runs with a temporary debug flag dumping full lidar — that diagnostic path lives outside the spec.
- **Metrics JSON extends Mission.md's six fields with `planner_error`** — Mission.md Phase 1 lists six fields (`time_to_goal, crashed, timed_out, path_length, mean_speed, wallclock_per_step`). The user explicitly chose the "planner_error" option in Phase 2 batch 2 question 3 to distinguish "tried and crashed" from "couldn't even start." Phase 1 ships the 7-field schema and we update Mission.md's Phase 4 description as part of T6 to match.
- **`time_to_goal` is `null` on crash/timeout/planner-error** — Mission.md Phase 4 calls it "undefined for crashes/timeouts." Phase 5's plot will filter on `!= null` for success-only distributions. Rejected: sim-time-at-failure (mixes success-time and failure-time on one axis) and -1 sentinel (gets accidentally averaged).
- **`path_length` = executed-trajectory length** — `Σ ‖state[t+1][:2] − state[t][:2]‖`. Works identically for path planners and (Phase 6) reactive planners; planned-path length wouldn't generalize. Rejected: planned-A*-path length (doesn't generalize), both-fields (inconsistent across algorithm files).
- **`mean_speed` = `path_length / sim_time`** — Over actual sim time, even on crash/timeout. Records what the robot did, not what the timeout allowed.
- **`wallclock_per_step` = mean of `EpisodeInfo.wallclock_per_step`** — Over all steps in the episode. Bonus telemetry per Mission.md Phase 4. NOT covered by TC15's byte-identity check (perf_counter readings are hardware-noise-dependent and cannot be byte-identical across two subprocess runs).
- **Planner failure → `planner_error: str` in JSON, all flags False, `time_to_goal: null`** — Distinguishes "could not even start" from "tried and crashed." Phase 5 folds it into failure rate; Phase 4 readers can tell them apart. No trace JSONL is written if the planner crashed before t = 0.
- **Runner catches `(ValueError, RuntimeError)` only — NOT bare `Exception`** — `manual_astar.astar_search` raises `RuntimeError`, `validate_start_and_goal` raises `ValueError`. Any other exception (TypeError, AttributeError, ImportError) is a programmer bug and must surface loudly. Catching bare Exception would mask import errors as `planner_error`.
- **JSON file is path-only self-identifying** — No `algorithm`/`seed`/`world` keys in the body. Smaller, cleaner. Phase 5 plot.py parses the path. (User explicitly chose this in Phase 2 batch 2 over self-describing JSON with redundant identifiers.)
- **Arena's 120 s timeout is the only termination cap** — `manual_astar.MAX_STEPS=1000` is ignored by the runner. Honouring both creates two failure modes for one event.
- **TC13 wall crash via teleport** — Reset, then set `arena._robot.state = [[20], [19], [π/2]]` (fallback to `._state` if read-only), clear the two sticky flags, then drive `v=1.0, w=0` for up to 100 steps. Wall B at center `(30, 20)` length 30 spans `x ∈ [15, 45]` and `y ∈ [19.6, 20.4]`. Pose `(20, 19, π/2)` puts the robot squarely under Wall B's span (x=20 well inside [15, 45]) with its center 0.6 m south of the wall's south face. Collision triggers when the robot center reaches y ≈ 19.4 — roughly 8 steps at v=1.0, dt=0.05. (The earlier draft used pose `(14, 19, π/2)`, which is 1 m WEST of the wall's left edge and would not collide. Corrected per critic.)
- **TC14 full goal arrival via the runner** — Subprocess-invoke `python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml --results-dir <tmpdir>`; parse the JSON and assert `time_to_goal != null and not crashed and not timed_out`; also load the trace JSONL and validate per-line schema + that the first record's `state == [2.0, 2.0, 0.0]` and the last record's `done is True`. Higher fidelity than TC8's `arrive_flag` injection.
- **TC15 determinism via subprocess + filecmp on TRACE JSONL ONLY** — Two clean subprocess invocations with the same seed write to two tempdirs; the two `<seed>.trace.jsonl` files must be byte-identical (`filecmp.cmp(a, b, shallow=False) is True`). The metrics JSON is NOT byte-compared because `wallclock_per_step` is a perf_counter mean and cannot be deterministic across runs. The two metrics JSONs ARE deep-compared on every field EXCEPT `wallclock_per_step` (load both, pop the field, assert equality).
- **TC16 planner-failure path** — `--world arena/arena_no_path.yaml --algorithm a_star_once --seed 0`. Asserts the metrics JSON exists, has `planner_error != null` (containing the substring "could not find a path"), and the trace JSONL does NOT exist.
- **TC13–TC16 live inline in `arena/arena.py`** — Same `TCi` pattern as the existing suite; no new test runner file. Keeps the harness's single source of truth in one place.
- **`results/` gitignored** — Phase 1 adds `results/*` + `!results/.gitkeep` to `.gitignore` and commits an empty `results/.gitkeep`. Generated data never enters git.
- **Phase 4 (Propose Approaches) skipped** — Architectural choices were enumerated as 2–3 options inside the Phase 2 question batches and the user picked one per batch. No further branching remains. Rejected alternatives are now documented inline in each Decision above (per critic axis-5 nit) so the spec is self-auditing.

## Acceptance Criteria

- [ ] **AC1:** `python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml` exits 0 and produces both `results/a_star_once/42.json` and `results/a_star_once/42.trace.jsonl`.
- [ ] **AC2:** The metrics JSON contains exactly the seven keys `time_to_goal`, `crashed`, `timed_out`, `path_length`, `mean_speed`, `wallclock_per_step`, `planner_error`. Type contract: `time_to_goal: float | null`, `crashed/timed_out: bool`, `path_length/mean_speed/wallclock_per_step: float`, `planner_error: str | null`.
- [ ] **AC3:** On a clean `a_star_once` run of `arena/arena_v1.yaml` at `seed=42`, the metrics satisfy:
  - `time_to_goal` is a float in `(50.0, 120.0)` — **gross-behavior smoke bound**; widen explicitly (do not silently pass) if `manual_astar` speed/grid constants are intentionally retuned;
  - `crashed == False`, `timed_out == False`;
  - `path_length > 64.0` — straight-line `(2,2)→(48,48)` is `65.05`; last waypoint may stop short within `FINAL_GOAL_TOLERANCE`;
  - `MIN_LINEAR_SPEED <= mean_speed <= MAX_LINEAR_SPEED` — **kinematic-consistency check**; bounds imported from `manual_astar` so they track controller retuning automatically. This is the regression-proof assertion;
  - `mean_speed > 0.5` — non-trivial motion (catches robot-stuck regressions);
  - `planner_error is None`.
- [ ] **AC4:** `python arena/arena.py arena/arena_v1.yaml --check` reports 17 PASS lines covering all 13 existing cases (TC1, TC2, TC2b, TC3..TC12) plus four new (TC13–TC16) and exits 0.
- [ ] **AC5:** TC13 — teleporting the robot to `(20, 19, π/2)` (using `arena._robot.state = ...` if writable; else falling back to `arena._robot._state = ...`), clearing both `collision_flag` and `arrive_flag`, then driving `v=1.0, w=0` produces `info.crashed == True` within 100 steps.
- [ ] **AC6:** TC14 — subprocess-invoking `run_episode` at `seed=42` against `arena/arena_v1.yaml` produces a metrics JSON whose contents satisfy AC3, AND the trace JSONL satisfies: every line is a valid JSON object with exactly the keys `{step, state, action, lidar_sha256, crashed, reached_goal, done}`; `state` is a 3-element list of floats; `lidar_sha256` is a 64-character hex string; the FIRST line has `step == 0`, `state == [2.0, 2.0, 0.0]`, `action == [0.0, 0.0]`, `done == False`; the LAST line has `done == True` and `reached_goal == True`. AC11 (programmatic `main()`) is verified via the same subprocess invocation — no separate in-process invocation.
- [ ] **AC7:** TC15 — invoking the runner twice with the same `--algorithm a_star_once --seed 42 --world arena/arena_v1.yaml` (in two separate tempdirs) produces:
  - byte-identical `<seed>.trace.jsonl` files (`filecmp.cmp(a, b, shallow=False) is True`);
  - metrics JSON files that are equal in every field EXCEPT `wallclock_per_step` (load both, delete the field, assert dict equality).
  `wallclock_per_step` is excluded because it is a `time.perf_counter` measurement and cannot be deterministic across two real-time runs.
- [ ] **AC8:** TC16 — `run_episode --world arena/arena_no_path.yaml --algorithm a_star_once --seed 0` exits 0 and produces a metrics JSON with `planner_error is not None` AND `"could not find a path"` appears in `planner_error` (the exact substring from `astar_search`'s RuntimeError message). All three flags are `False`, `time_to_goal is None`. The `<seed>.trace.jsonl` file does NOT exist.
- [ ] **AC9:** `results/.gitkeep` is tracked; `.gitignore` contains the entries `results/*` and `!results/.gitkeep`. Verifiable: `git check-ignore -q results/a_star_once/sample.json` returns 0 (i.e., the path would be ignored).
- [ ] **AC10:** `from planners import AStarOncePlanner, PathPlanner, Path` succeeds in a Python REPL. `isinstance(AStarOncePlanner(), PathPlanner) is True`, `AStarOncePlanner().name == "a_star_once"`, and `callable(AStarOncePlanner().plan) is True`. (The runtime_checkable Protocol only verifies attribute presence — these three assertions tighten the contract to value + callability.)
- [ ] **AC11:** `from runners.run_episode import main` succeeds; `main([...])` is callable. (Verified end-to-end inside AC6's subprocess; no separate AC11-only invocation, to avoid doubling `--check`'s wallclock.)
- [ ] **AC12:** `CLAUDE.md` gains a new "Phase 1 — Episode runner" section documenting (a) the run command, (b) the `results/<algo>/<seed>.{json,trace.jsonl}` layout, (c) the 7-field metrics JSON schema (noting it extends Mission.md's 6-field list with `planner_error`), (d) the trace JSONL schema, (e) one-line summaries of TC13–TC16, (f) the new `arena/arena_no_path.yaml` fixture.

## Data Model

```python
# planners/_types.py
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np

Path = tuple[np.ndarray, ...]  # ordered (2,)-shaped float64 world-frame waypoints; last == goal


@runtime_checkable
class PathPlanner(Protocol):
    name: str  # e.g. "a_star_once" — must match the results/<name>/ subdir

    def plan(
        self,
        world_yaml: str,
        initial_dynamic_snapshot: tuple,  # () in Phase 1 / Phase 0; tuple of states in Phase 2
        lidar0: np.ndarray,               # (360,) float64 from arena.reset(); may be all-NaN if reset's lidar_status == "missing". Static planners (A*) ignore it; reactive planners (Phase 6) use it.
    ) -> Path: ...
```

```python
# planners/a_star.py
from manual_astar import plan_waypoints

class AStarOncePlanner:
    name = "a_star_once"

    def plan(self, world_yaml, initial_dynamic_snapshot, lidar0):
        _, _, _, waypoints = plan_waypoints(world_yaml)
        return tuple(waypoints)
```

```jsonc
// results/<algorithm>/<seed>.json — keys are stable; new fields can be added but never renamed.
// Extends Mission.md Phase 1's six-field list with a seventh `planner_error` field; Mission.md is updated in T6.
{
  "time_to_goal":        12.345,    // float seconds on success, null on crash/timeout/planner_error
  "crashed":             false,
  "timed_out":           false,
  "path_length":         67.890,    // sum of consecutive ‖state[t+1][:2] − state[t][:2]‖
  "mean_speed":          0.876,     // path_length / sim_time (0.0 if sim_time == 0)
  "wallclock_per_step":  0.0123,    // mean of EpisodeInfo.wallclock_per_step over all steps; NOT byte-deterministic across runs
  "planner_error":       null       // str message if plan() raised; null otherwise
}
```

```jsonc
// results/<algorithm>/<seed>.trace.jsonl — one JSON object per line; written only if planning succeeded.
// json.dumps(sort_keys=True, separators=(",", ":")) for stable line-level diffs.
// Step 0 records the post-reset state with action=[0.0, 0.0] sentinel; subsequent steps record the state AFTER each arena.step(action) call.
{"action":[0.0,0.0],"crashed":false,"done":false,"lidar_sha256":"<64-hex>","reached_goal":false,"state":[2.0,2.0,0.0],"step":0}
{"action":[1.0,0.0],"crashed":false,"done":false,"lidar_sha256":"<64-hex>","reached_goal":false,"state":[2.05,2.0,0.0],"step":1}
// ... continuing until done == True on the final line.
```

## API Contracts (CLI)

```
python -m runners.run_episode \
    --algorithm <name>      # required; matches a registered planner (Phase 1: only "a_star_once")
    --seed <int>            # required
    --world <yaml_path>     # required; e.g. arena/arena_v1.yaml or arena/arena_no_path.yaml
    [--render]              # optional; default False
    [--results-dir <dir>]   # optional; default "results"

Writes:
  <results-dir>/<algorithm>/<seed>.json          # always written
  <results-dir>/<algorithm>/<seed>.trace.jsonl   # written only if plan() succeeded

Exit codes:
  0  — episode terminated (success, crash, timeout, or planner failure all return 0; the JSON disambiguates)
  2  — malformed CLI args, unknown algorithm, missing world file, or Arena __init__ failure (e.g., bad lidar config)
```

Programmatic entry: `runners.run_episode.main(argv: list[str] | None = None) -> int`.

## Error Handling

- **Planner raises `ValueError` or `RuntimeError` during `plan()`** (the only two exception classes the existing `manual_astar` code paths produce — `astar_search` → `RuntimeError`, `validate_start_and_goal` / `parse_obstacle` → `ValueError`): caught, written as `planner_error: <str(exc)>` in the metrics JSON; all flags False; `time_to_goal: null`. Trace JSONL is NOT written. Exit code 0.
- **Planner raises any OTHER exception** (TypeError, AttributeError, ImportError, etc.): re-raise. These are programmer bugs and must surface loudly, not be silently bucketed as `planner_error`.
- **Arena raises `ArenaConfigError` during `__init__`** (e.g., missing lidar block, malformed YAML): NOT caught — propagate, exit code 2, no files written, error logged to stderr. This is a config-level failure mode distinct from a planner-level failure.
- **Mid-episode `ArenaRuntimeError`**: re-raise; arena is closed in a `finally` block; no metrics written. Harness bugs must be loud.
- **Action-validation `ValueError` from `arena.step`**: re-raise. Runner-side bug; must be loud.
- **Disk-full / write error during JSON or JSONL write**: re-raise (no fallback). Arena closed in `finally`.
- **KeyboardInterrupt**: arena closed in `finally`; no metrics or trace written; non-zero exit.
- **Unknown `--algorithm`**: argparse rejects with exit code 2. Algorithm registry is a dict `{"a_star_once": AStarOncePlanner}` in `runners/run_episode.py` for now; Phase 6 expands it.

## Testing Strategy

**Levels:** Inline TCi suite in `arena/arena.py --check` (existing pattern). All new TCs use the same `assert + clear message` convention as TC1–TC12.

| ID    | Test Case                                                            | Type            | Expected Behavior                                                                                                                  |
|-------|----------------------------------------------------------------------|-----------------|------------------------------------------------------------------------------------------------------------------------------------|
| TC13  | Wall crash via teleport to `(20, 19, π/2)` + straight drive          | Inline          | `info.crashed == True` and `done == True` within 100 steps                                                                         |
| TC14  | Full A* drive through the runner (subprocess) + trace content audit  | Integration     | Metrics JSON satisfies AC3; trace JSONL passes per-line schema check; first record is `(2,2,0)` with zero action; last record is `done=True, reached_goal=True` |
| TC15  | Determinism — two subprocess runs of seed=42, byte-diff outputs      | Integration     | `filecmp.cmp(jsonl_a, jsonl_b, shallow=False) is True`; metrics JSONs equal in every field EXCEPT `wallclock_per_step`             |
| TC16  | Planner failure path on `arena/arena_no_path.yaml`                   | Integration     | Metrics JSON has `planner_error is not None` and contains "could not find a path"; all flags False; trace JSONL does NOT exist     |

Existing TC1–TC12 (including TC2b) must continue to PASS unchanged after Phase 1's edits.

**Test data:** TC14/TC15/TC16 use `tempfile.TemporaryDirectory()` for the `--results-dir` to avoid polluting the repo's `results/` during a `--check` run. TC15 creates two separate tempdirs and compares both files. TC16 uses a tempdir too. **Expected `--check` wallclock budget post-Phase-1**: roughly TC1–TC12 baseline + 100 sim steps for TC13 (~few seconds) + 3 full a_star_once episodes for TC14/TC15/TC16 (~70 s of sim time each, but irsim runs faster-than-real-time headless — measure during implementation and document in CLAUDE.md if it exceeds ~3 min).

**Run command:** `python arena/arena.py arena/arena_v1.yaml --check` (must report 17 PASS, exit 0). Plus a smoke-style integration run: `python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml` followed by inspection of `results/a_star_once/42.json`.

## Tasks

| ID   | Task                                              | Blocked By | Risk | Files                                                      | Description |
|------|---------------------------------------------------|------------|------|------------------------------------------------------------|-------------|
| T1   | Define `Path` + `PathPlanner` Protocol            | —          | low  | `planners/_types.py`                                       | Create `planners/` package directory. In `_types.py`: `Path = tuple[np.ndarray, ...]` and `@runtime_checkable class PathPlanner(Protocol)` with `name: str` attribute and `def plan(self, world_yaml: str, initial_dynamic_snapshot: tuple, lidar0: np.ndarray) -> Path: ...`. Does NOT touch `planners/__init__.py` — that file is owned entirely by T2 to avoid the two-tasks-one-file conflict. Satisfies AC10 (partial: types exist). |
| T2   | Implement `AStarOncePlanner` + assemble package   | T1         | med  | `planners/a_star.py`, `planners/__init__.py`               | Write `planners/__init__.py` from scratch: `from planners._types import Path, PathPlanner; from planners.a_star import AStarOncePlanner; __all__ = ["Path", "PathPlanner", "AStarOncePlanner"]`. Then in `a_star.py`: class `AStarOncePlanner` with `name = "a_star_once"` and `plan(self, world_yaml, initial_dynamic_snapshot, lidar0)` returning `tuple(manual_astar.plan_waypoints(world_yaml)[3])`. Let exceptions propagate (runner catches them). No mutation of `manual_astar` globals. Satisfies AC10. |
| T3   | Implement `runners/run_episode.py`                | T2         | high | `runners/__init__.py`, `runners/run_episode.py`            | Create `runners/__init__.py` as an empty file (one-line comment `# intentional` is acceptable). Argparse CLI matching the API Contracts section. `def main(argv=None) -> int`. Builds `ALGORITHMS = {"a_star_once": AStarOncePlanner}`. Constructs `Arena(world, seed, render=args.render)`. Calls `state0, lidar0, _ = arena.reset()`. **Write the step-0 trace line immediately** with `step=0, state=state0.tolist(), action=[0.0, 0.0], lidar_sha256=sha256(lidar0.tobytes()).hexdigest(), crashed=False, reached_goal=False, done=False`. Then `try: waypoints = planner.plan(world, arena.initial_dynamic_snapshot, lidar0) except (ValueError, RuntimeError) as exc: close trace file (delete it, since the spec contract is "no trace if planner failed"), write metrics JSON with planner_error=str(exc), close arena, return 0`. On success: `follower = WaypointFollower(list(waypoints), WAYPOINT_REACHED_DISTANCE)`; loop `compute_action → arena.step(action) → write trace line` until `done`. Each loop iteration: write JSONL line with `step=info.step_idx, state=state.tolist(), action=action.flatten().tolist(), lidar_sha256, crashed=info.crashed, reached_goal=info.reached_goal, done=done`. Accumulate `path_length += ‖prev_xy − state[:2]‖` (prev_xy starts as state0[:2]). Sum wallclock. After loop: `time_to_goal = info.sim_time if info.reached_goal else None`; `mean_speed = path_length / info.sim_time if info.sim_time > 0 else 0.0`; `wallclock_per_step = total_wallclock / max(step_count, 1)`. Write metrics JSON. Always `arena.close()` in `finally`. Ensure trace file is `flush() + os.fsync(fileno())` before close. Use `pathlib.Path.mkdir(parents=True, exist_ok=True)` for the output dir. Satisfies AC1, AC2, AC3, AC11. |
| T4a  | TC13 — wall crash via teleport                    | —          | med  | `arena/arena.py`                                           | Append `def tc13(yaml_path, seed): ...` plus registration in `_run_checks`. Body: `arena = Arena(yaml_path, seed); try: arena.reset(); try: arena._robot.state = np.array([[20],[19],[np.pi/2]], dtype=float); except (AttributeError, TypeError): arena._robot._state = np.array([[20],[19],[np.pi/2]], dtype=float); arena._robot.collision_flag = False; arena._robot.arrive_flag = False; action = np.array([[1.0],[0.0]], dtype=float); for up to 100 steps: step; if done and info.crashed: return; raise AssertionError("TC13 did not crash within 100 steps")`. Runs independently of T2/T3 (no runner dependency). Satisfies AC5. |
| T4b  | TC14/TC15/TC16 — runner integration checks        | T3         | high | `arena/arena.py`                                           | Append `tc14`, `tc15`, `tc16` and register all three. Each uses `subprocess.run([sys.executable, "-m", "runners.run_episode", ...], check=False, cwd=repo_root)` then loads the resulting JSON / JSONL. **TC14:** seed=42, world=arena_v1.yaml; assert AC3 contract on JSON; iterate trace JSONL lines and verify each has the 7 required keys with right types; first line is `{step:0, state:[2.0,2.0,0.0], action:[0.0,0.0], done:False, reached_goal:False, crashed:False, lidar_sha256: <64-hex>}`; last line has `done:True, reached_goal:True`. **TC15:** invoke runner twice into two separate tempdirs (seed=42 both times); `filecmp.cmp(jsonl_a, jsonl_b, shallow=False)` must be True; load both metrics JSONs, pop `wallclock_per_step`, assert dict equality. **TC16:** seed=0, world=`arena/arena_no_path.yaml`; assert JSON has `planner_error is not None` and `"could not find a path" in planner_error`, all three flags False, `time_to_goal is None`, and `os.path.exists(<seed>.trace.jsonl)` is False. Repo-root path detection mirrors TC10's existing `sys.path` snippet. Satisfies AC4, AC6, AC7, AC8. **Higher-risk than T4a because subprocess flakes and tempdir handling are the main failure surface.** |
| T5   | Gitignore `results/` and commit `.gitkeep`        | —          | low  | `.gitignore`, `results/.gitkeep`                           | Append two lines to `.gitignore`: `results/*` and `!results/.gitkeep`. Create an empty `results/.gitkeep`. If `.gitignore` does not exist, create it with these two lines plus a leading comment line `# Generated experiment data — see Mission.md Phase 1`. **Verification step** (do this as part of the task): run `git check-ignore -q results/a_star_once/sample.json; echo $?` (or `$LASTEXITCODE` in PowerShell) — must be `0` (meaning the file WOULD be ignored). Also document in the task notes: ignore rules do NOT retroactively un-track files already committed. Satisfies AC9. |
| T6a  | Author `arena/arena_no_path.yaml` fixture         | —          | low  | `arena/arena_no_path.yaml`                                 | Copy `arena/arena_v1.yaml` verbatim (preserve `world`, `robot`, the lidar block, start `[2,2,0]`, goal `[48,48,0]`). Replace the `obstacle[]` list with a tight rectangular enclosure around the start, e.g. four rectangles forming a 1.5 m × 1.5 m box centered at `(2, 2)`: top wall at `(2, 3, 0)` length 1.5 width 0.2; bottom wall at `(2, 1, 0)` length 1.5 width 0.2; left wall at `(1, 2, π/2)` length 1.5 width 0.2; right wall at `(3, 2, π/2)` length 1.5 width 0.2. After inflation (robot_radius 0.2 + SAFETY_MARGIN 0.15 = 0.35), the box's interior is fully sealed — A* will raise `RuntimeError("A* could not find a path…")`. **Verify before merging T4b** that `python arena/arena.py arena/arena_no_path.yaml --check` runs TC1–TC12 successfully (TC10/TC11 may need to be made tolerant of obstacle-count differences — if so, document the workaround; the simplest path is to run TC16 against this fixture WITHOUT first running the full --check suite on it, since TC11 hard-codes 14 obstacles for arena_v1.yaml). Satisfies prerequisite for AC8 / TC16. |
| T6b  | Update `CLAUDE.md` + `Mission.md` with Phase 1    | —          | low  | `CLAUDE.md`, `Mission.md`                                  | In `CLAUDE.md`, add a new section titled `## The episode runner (Phase 1)` immediately after the existing `## The arena harness (Phase 0)` section. Document: the `python -m runners.run_episode` command with flags, the `results/<algorithm>/<seed>.{json,trace.jsonl}` layout, the 7-field metrics JSON schema, the trace JSONL schema, the determinism guarantee (same seed → byte-identical trace, near-identical metrics), one-line summaries of TC13–TC16, and the `arena/arena_no_path.yaml` fixture's role. Mirror the existing Phase 0 section's prose style. In `Mission.md` Phase 1 line 67, update the metrics-fields list to seven fields (add `planner_error`). **Re-review T6b after T3 + T4b land** — if any runner/CLI detail shifted during implementation, update the docs to match. Satisfies AC12. |

**Parallelism:**
- T1, T4a, T5, T6a, T6b can start immediately (no blockers).
- T2 waits on T1.
- T3 waits on T2.
- T4b waits on T3 and T6a (T6a creates the fixture TC16 needs).
- T6b re-review after T3 + T4b land is a low-cost doc patch, not a re-task.

## Notes for Implementer

- **`arena._robot.state` setter (TC13)**: in irsim 2.9.3 `robot.state` is a `(3, 1)` numpy column vector. Assigning `arena._robot.state = np.array([[20],[19],[np.pi/2]], dtype=float)` should work, but the public attribute MAY be read-only depending on irsim's internal property setup. Fallback: `arena._robot._state = ...` (the private backing field). T4a wraps the assignment in a try/except to handle both cases. After teleport, you MUST clear both `collision_flag` and `arrive_flag` (the same defensive pattern `Arena.reset()` already does at line 92–94).
- **Why `(20, 19, π/2)` for TC13**: `θ = π/2` means facing `+y`. Wall B is the rectangle at center `(30, 20)` with `length=30, width=0.8`, spanning `x ∈ [15, 45]` and `y ∈ [19.6, 20.4]`. The pose `(20, 19, π/2)` places the robot squarely under Wall B's span (x=20 is well inside [15, 45]) with its center 0.6 m south of the wall's south face at y=19.6. With robot radius 0.2, collision occurs when the robot center reaches y ≈ 19.4 — roughly 8 steps at v=1.0, dt=0.05, well within the 100-step budget. Pillars `(12, 8)` and `(8, 25)` are NOT on the line x=20.
- **Earlier TC13 design was wrong**: an earlier draft used pose `(14, 19, π/2)`. That pose places the robot 1 m WEST of Wall B's left edge (x=15) and a straight +y drive never intersects the wall. Corrected per critic review.
- **irsim's arrive_flag tolerance vs. `compute_action`'s FINAL_GOAL_TOLERANCE=0.1**: `compute_action` zeros linear velocity once `WaypointFollower.is_finished` and `distance < 0.1`. irsim's default `arrive_threshold` is NOT set in `arena_v1.yaml`. If irsim's default is *larger* than 0.1, `arrive_flag` fires on its own once we get within its tolerance, fine. If irsim's default is *smaller* than 0.1, the robot decelerates to 0 at distance 0.1 from `(48, 48)` and `arrive_flag` NEVER fires — TC14 will timeout instead of succeed. **Mitigation**: during T3 implementation, dry-run a single episode and verify `info.reached_goal == True` at termination; if not, either (a) shrink `manual_astar.FINAL_GOAL_TOLERANCE` to a value certainly below irsim's default, or (b) add `goal_threshold: 0.3` (or whatever irsim's parameter is) to `arena/arena_v1.yaml`'s robot block. Document the resolution in T6b.
- **Lidar hash**: `import hashlib; hashlib.sha256(lidar.tobytes()).hexdigest()`. NumPy's `tobytes()` is endianness-aware and NaN-bit-stable — the same NaN value hashes the same. Use `np.float64` consistently (Arena already does). When `lidar_status == "missing"`, the array is all-NaN `(360,)`; its tobytes() hash is a well-defined constant. The planner ignores `lidar0`, so a NaN-filled first-step lidar does not affect planning.
- **JSONL line format**: `json.dumps({...}, sort_keys=True, separators=(",", ":"))` followed by `"\n"`. Compact and order-stable. `separators=(",", ":")` removes the post-`,`/post-`:` space — needed for the byte-identity check in TC15.
- **Trace fsync (TC15)**: between `file.close()` and the second subprocess run, the OS page cache could in principle hold dirty pages that get flushed in a different order across runs. Add explicit `file.flush(); os.fsync(file.fileno())` before close inside `run_episode` to remove any doubt.
- **`time_to_goal == null` in JSON**: `json.dumps({"time_to_goal": None})` produces `"time_to_goal": null` natively — no custom encoder needed.
- **`path_length` on early termination**: still record the actual traveled distance. Don't null it. Same for `mean_speed` (compute over `info.sim_time` whatever it ended up being).
- **`mean_speed == 0.0` edge case**: if `info.sim_time == 0` (zero steps taken — shouldn't happen because the runner always steps once before checking done, but be defensive), set `mean_speed = 0.0`.
- **Subprocess + sys.path (TC14–TC16)**: subprocesses inherit the parent's environment but NOT its `sys.path` modifications. Use `cwd=repo_root` and `sys.executable` so the child resolves `runners.run_episode` against the repo's top-level. The existing TC10 snippet (`repo_root = str(Path(__file__).resolve().parent.parent)`) is the pattern to copy.
- **`runners/__init__.py`** is a single-line file `# intentional` (or empty). The runner is invoked as a script via `python -m runners.run_episode`, which only requires the package to be importable.
- **The `PathPlanner` Protocol carries `lidar0` in its signature**: A* ignores it (waypoints are derived from the static map). DWA / APF / D* Lite (all Phase 6) will use it. Adding the parameter now means Phase 6 doesn't change the interface for path planners. Reactive planners get a separate `ReactivePlanner` Protocol in Phase 6 returning `np.ndarray (2, 1)` actions instead of a `Path`; the harness dispatches on output type per Mission.md line 145.
- **`manual_astar.run_simulation` stays**: we are not deleting it. It remains the loose-style standalone demo.
- **`MAX_STEPS = 1000` in manual_astar.py**: the runner does NOT use this constant. The loop terminates on `Arena.done`. Arena's 120 s timeout (~2400 steps at `dt = 0.05`) is the only step cap.
- **`results/` directory creation**: `run_episode` does `Path(args.results_dir, args.algorithm).mkdir(parents=True, exist_ok=True)` before opening either output file. No surprises if `results/` was deleted.
- **`results/.gitkeep` (T5)**: must be empty (0 bytes) and committed. The `.gitignore` pattern `results/*` would otherwise hide the directory entirely from git. **Note**: ignore rules added now do NOT retroactively un-track files already committed — if a developer accidentally `git add`-ed `results/a_star_once/42.json` before T5 lands, they must `git rm --cached` it separately.
- **`CLAUDE.md` + `Mission.md` edits (T6b)**: insert the new section between the existing Phase 0 section and "## Conventions worth preserving" so the file remains chronologically structured. Update Mission.md Phase 1's metrics list to seven fields. Re-review after T3/T4b to ensure docs match the as-built CLI.
- **`arena/arena_no_path.yaml` and TC10/TC11**: TC10 (`manual_astar.build_occupancy_grid` + `validate_start_and_goal`) will RAISE on the no-path fixture because the start is enclosed → `validate_start_and_goal` succeeds (start not blocked) but `astar_search` later raises. TC11 hard-codes `14 obstacles, 2 rectangles, 12 circles` — that's specific to arena_v1.yaml and would fail on arena_no_path.yaml. **Therefore**: T4b's TC16 invokes the runner directly against `arena/arena_no_path.yaml` and does NOT route through `--check` on that file. The `--check` CLI defaults to `arena_v1.yaml` for the full suite (`python arena/arena.py arena/arena_v1.yaml --check` continues to be the canonical command).
- **Rollback plan**: every Phase 1 file is new except `arena/arena.py`, `CLAUDE.md`, `Mission.md`, and `.gitignore`. To revert: delete `planners/`, `runners/`, `results/`, `arena/arena_no_path.yaml`, the four new TC functions in `arena/arena.py`, the new section in `CLAUDE.md`, the Mission.md edit, and the two `results/` lines in `.gitignore`. No data migration, no schema changes.
