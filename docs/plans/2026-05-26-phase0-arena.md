# Phase 0 — Arena Plan

**Goal:** Ship a reusable, seeded 50×50 test environment (`arena/arena_v1.yaml` + `arena/arena.py`) that every planner in the Mission.md experiment will run against on identical conditions. Static obstacles only; dynamic traffic is Phase 2.

**Approach:** A thin `Arena` class wrapping `irsim.make()` and exposing `reset()` / `step(action) -> (state, lidar, done, info)` plus an `EpisodeInfo` dataclass. World layout lives in a single canonical YAML using only irsim's `rectangle` (walls) and `circle` (pillars) primitives so the existing `manual_astar.py` planner can consume it untouched. `numpy.random.SeedSequence` is plumbed now (master seed → child streams for traffic and motion, both unused in Phase 0) so the Phase 2 spawner can attach without an interface change. An opt-in `render=True` flag and `python arena/arena.py <yaml>` smoke-test `__main__` let you eyeball the YAML and confirm the lifecycle works end-to-end.

## Scope

- **In scope:**
  - `arena/__init__.py` re-exporting `Arena` and `EpisodeInfo`.
  - `arena/arena_v1.yaml` — canonical 50×50 world with 2 rectangle walls + 12 circle pillars, robot at `(2, 2)` with goal `(48, 48)`, 360-beam 10 m lidar, robot radius 0.2 m.
  - `arena/arena.py` — `Arena` class (`reset`, `step`, `close`, `initial_dynamic_snapshot` property), `EpisodeInfo` frozen dataclass, smoke-test `__main__`.
  - Update to project `CLAUDE.md` documenting the `arena/` module and the smoke-test command.
- **Out of scope:**
  - `DynamicObstacle` / `TrafficSpawner` — Phase 2. (Mission.md Phase 0 says "The `arena` module is responsible for spawning the dynamic-obstacle population from a seeded RNG"; we interpret that as **Phase 0 owns the seam, Phase 2 implements the spawner behind it**. The `initial_dynamic_snapshot` property and the `traffic_rng` / `motion_rng` plumbing are the seam.)
  - Per-episode metrics writer (`results/<algorithm>/<seed>.json`) — Phase 1.
  - Reproducibility regression test (run-seed-twice-diff-JSONL) — Phase 1/3.
  - Any planner implementation or planner-adapter glue — Phase 6.
  - `results/plot.py` — Phase 5.

## Decisions

- **Static-only Phase 0, no spawner code** — keeps the PR small and lets us land the API contract before the spawner design churns. The `initial_dynamic_snapshot` property returns `()` now; in Phase 2 it returns a tuple of dynamic-obstacle states. No signature change required (the return is `tuple[Any, ...]` with `Any` narrowed in Phase 2).
- **Two staggered horizontal walls of length 30 m (axis-aligned rectangles), 12 circular pillars (r = 0.6 m)** — Wall A at center `(20, 30)` spans `x ∈ [5, 35]`; Wall B at center `(30, 20)` spans `x ∈ [15, 45]`. Walls are offset so the only short route from `(2, 2)` to `(48, 48)` is a forced zig-zag: through Wall B's left gap (`x < 14.65` after inflation) at `y ≈ 20`, then through Wall A's right gap (`x > 35.35` after inflation) at `y ≈ 30`. This honors Mission.md's "long walls forcing corridors and choke points that Phase 2 traffic will naturally pinch." Axis-aligned (theta = 0) for legibility.
- **Lidar: 360 beams, 2π rad, range_max = 10 m** — at 1.5× robot top speed (1.5 m/s), 10 m gives ~6.6 s of advance warning, enough for DWA / APF / replanning grid planners to react. 1° angular resolution keeps DWA happy.
- **Robot params unchanged from `manual_astar.py`** — radius = 0.2 m, top speed = 1.0 m/s. Reuses existing controller tuning untouched and gives a ~50 s baseline straight-line run vs the 120 s timeout (comfortable headroom).
- **`Arena` class, not gymnasium env** — this is not RL; injecting a `reward` channel is misleading. Plain `(state, lidar, done, info)` matches Mission.md's signature.
- **`SeedSequence` plumbed now even though unused** — `master_seed → ss.spawn(2)` for `traffic_rng` and `motion_rng`. Both rngs are constructed in `__init__`, neither is read in Phase 0. Phase 2 hooks the spawner onto `traffic_rng` and motion onto `motion_rng`; the rng state at `t = 0` is already deterministic, so adding consumers later does not retroactively change earlier seeds.
- **`render` is a constructor flag, default `False`** — batch sweep (Phase 4) wants headless; manual debug wants visible.
- **`done` is sticky; `step()` after `done` raises `RuntimeError`** — silent no-op masks caller bugs (e.g., harness forgot to `reset()` between episodes).
- **Timeout enforced by `arena.py` at `sim_time >= 120.0 s`** — irsim has no built-in episode timer. `sim_time = step_idx * dt` where `dt` comes from the loaded irsim env (pinned by T0).
- **Walls = rectangles, pillars = circles** — `manual_astar.py`'s existing `ObstacleSpec` handles both natively.
- **Corridor passages ≥ 3 m wide after inflation** — wide enough that no planner is artificially squeezed in Phase 0; Phase 2 dynamic obstacles supply the real difficulty.
- **Action shape is `np.ndarray` shape `(2, 1)` dtype float `[[v], [w]]`** — matches `manual_astar.py:512` and `manual_obstacle.py`'s action format. Passed to irsim as `env.step([action])` (list-wrapped).

## Acceptance Criteria

- [ ] AC1: `python arena/arena.py arena/arena_v1.yaml --render` opens a 50×50 irsim window showing two horizontal walls, 12 circular pillars, the robot at `(2, 2)`, and the goal marker at `(48, 48)`.
- [ ] AC2: `from arena import Arena, EpisodeInfo` works without error.
- [ ] AC3: `Arena('arena/arena_v1.yaml', seed=42)` constructs without error.
- [ ] AC4: `state, lidar, info = arena.reset()` returns `state.shape == (3,)`, `lidar.shape == (360,)`, `lidar.dtype == np.float64`, and `info` is an `EpisodeInfo` instance with all fields listed in the Data Model below. When the lidar sensor returns no scan, `lidar` is an all-NaN `(360,)` array and `info.lidar_status == "missing"`.
- [ ] AC5: `state, lidar, done, info = arena.step(np.array([[0.0], [0.0]], dtype=float))` returns `done == False` and increments `info.step_idx` by 1 and `info.sim_time` by `dt`.
- [ ] AC6: Driving constant `np.array([[1.0], [0.3]], dtype=float)` (forward + slow left turn) from start eventually produces `info.crashed == True` and `done == True` within 200 steps. Rationale: with theta=0 at `[2, 2]`, a constant 0.3 rad/s left turn at v=1.0 m/s traces a circular arc of radius ~3.33 m centered near `(2, 5.33)`. Pillar `(5, 5)` lies inside that arc, so the robot collides with it after ~97 steps at `dt ≈ 0.05`.
- [ ] AC7: Looping `arena.step(np.array([[0.0], [0.0]], dtype=float))` past `sim_time >= 120.0` produces `info.timed_out == True` and `done == True`.
- [ ] AC8: Calling `step()` after `done == True` raises `RuntimeError`.
- [ ] AC9: `arena.reset()` after a finished episode returns `info.sim_time == 0.0`, `info.step_idx == 0`, all flags `False`. Evaluated against the path (A or B) selected by T0; under PATH B the test additionally asserts that the rebuilt env still exposes the same `dt` and robot handle shape as the original.
- [ ] AC10: Reaching the goal (asserted by directly setting the robot pose near `(48, 48)` and stepping once, OR by injecting `robot.arrive_flag = True` and stepping once with a zero action) produces `info.reached_goal == True` and `done == True`.
- [ ] AC11: `arena_v1.yaml` loads cleanly via `manual_astar.py`'s `load_world(...)`; the full call sequence `world = load_world(path); grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN); validate_start_and_goal(world, grid)` succeeds — i.e., `(2, 2)` and `(48, 48)` are unblocked after `robot.radius + SAFETY_MARGIN` inflation.
- [ ] AC12: The project `CLAUDE.md` has a new entry describing `arena/` and the smoke-test command.

## Data Model

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class EpisodeInfo:
    sim_time: float                   # seconds since reset
    step_idx: int
    crashed: bool                     # irsim collision_flag
    timed_out: bool                   # sim_time >= timeout_s
    reached_goal: bool                # irsim arrive_flag
    distance_to_goal: float           # euclidean, world units
    wallclock_per_step: float         # last step's perf_counter delta
    dynamic_obstacle_count: int       # always 0 in Phase 0
    lidar_status: str                 # "ok" | "missing"
```

```python
class Arena:
    def __init__(
        self,
        yaml_path: str | Path,
        seed: int,
        render: bool = False,
        timeout_s: float = 120.0,
    ) -> None: ...

    def reset(self) -> tuple[np.ndarray, np.ndarray, EpisodeInfo]: ...

    def step(
        self,
        action: np.ndarray,        # shape (2, 1) dtype float: [[v], [w]]
    ) -> tuple[np.ndarray, np.ndarray, bool, EpisodeInfo]: ...

    @property
    def initial_dynamic_snapshot(self) -> tuple[Any, ...]:
        """Snapshot of dynamic obstacles at t=0. Empty in Phase 0; Phase 2 narrows the type."""

    def close(self) -> None: ...
```

## API Contracts

`step(action)` semantics:

```
Input:  action: np.ndarray shape (2, 1) dtype float -> [[v], [w]]
                v   in [-MAX_LINEAR_SPEED,  MAX_LINEAR_SPEED]   (clipped by caller, not validated for range)
                w   in [-MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED]  (clipped by caller, not validated for range)

Output: (state, lidar, done, info)
    state: np.ndarray shape (3,) dtype float64 -> [x, y, theta]
           (Note: irsim's robot.state is shape (3, 1); Arena flattens via [:, 0] before returning.)
    lidar: np.ndarray dtype float64 shape (360,)
           Produced by calling robot.get_lidar_scan() and extracting 'ranges':
             - If get_lidar_scan() returns a falsy value (no scan this tick):
                 return np.full((360,), np.nan, dtype=float64)
                 set info.lidar_status = "missing"
             - Else convert scan['ranges'] via np.asarray(..., dtype=float64).
                 If len != 360, raise ArenaConfigError at Arena.__init__ time
                   (validated once against robot.sensors.lidar2d.number, NOT per step).
                 If 'ranges' key missing on a non-falsy scan, raise ArenaRuntimeError.
                 Entries are meters in [range_min, range_max]; substitute np.nan for any
                 entry >= range_max or non-finite from the sensor.
             - When scan present and well-formed, set info.lidar_status = "ok".
    done : bool                    -> info.crashed OR info.timed_out OR info.reached_goal
    info : EpisodeInfo             -> see Data Model

Errors:
    RuntimeError:     step() called when self._done is True (caller must reset)
    ValueError:       action shape != (2, 1), or action contains NaN/Inf, or action dtype is non-float
    ArenaRuntimeError: get_lidar_scan() returned a dict without 'ranges' key mid-episode

irsim call: env.step([action])  (list-wrapped per manual_astar.py:543 and manual_obstacle.py)
```

`reset()` semantics:

```
Input:  none
Output: (state, lidar, info)
    state: np.ndarray shape (3,) dtype float64 -> robot start pose from YAML, flattened from (3,1)
    lidar: np.ndarray shape (360,) dtype float64 -> initial lidar scan (same extraction as step)
    info : EpisodeInfo             -> sim_time=0.0, step_idx=0, all flags False
                                       distance_to_goal computed, wallclock_per_step=0.0,
                                       dynamic_obstacle_count=0, lidar_status set from scan

Side effects:
    Re-derives traffic_rng and motion_rng from master_seed via SeedSequence.spawn(2).
    Both rngs are unused in Phase 0 but rebuilt deterministically.

Implementation path: selected by T0.
    PATH A (native reset): env.reset() exists and restores robot to YAML start pose, zeros
        internal time/step counters, preserves the render window. reset() calls env.reset().
    PATH B (rebuild reset): env.reset() missing or insufficient. reset() calls env.end() then
        irsim.make(yaml_path, display=render) again; dt and robot handle re-cached after
        rebuild. Render window flickers per reset.
```

`arena_v1.yaml` schema (concrete — pending T1 layout finalization):

```yaml
world:
  height: 50
  width: 50

robot:
  kinematics: {name: 'diff'}
  shape: {name: 'circle', radius: 0.2}
  state: [2, 2, 0]
  goal:  [48, 48, 0]
  color: 'g'
  sensors:
    - name: 'lidar2d'
      range_min: 0.05
      range_max: 10.0
      angle_range: 6.283185307179586   # 2*pi
      number: 360
      alpha: 0.2                       # opacity; drop if irsim parser rejects (verified in T0)
  plot:
    show_trajectory: True
    show_goal: True

obstacle:
  # Wall A — long horizontal, upper. Spans x in [5, 35] at y ~ 30.
  - shape: {name: 'rectangle', length: 30, width: 0.8}
    state: [20, 30, 0]

  # Wall B — long horizontal, lower. Spans x in [15, 45] at y ~ 20.
  - shape: {name: 'rectangle', length: 30, width: 0.8}
    state: [30, 20, 0]

  # 12 pillars (r = 0.6 m). Positions chosen so:
  #  - Start (2,2) and goal (48,48) clear of inflation (>= 4 m to nearest pillar).
  #  - The natural zig-zag (left gap of Wall B -> right gap of Wall A) stays passable.
  #  - The inter-wall corridor (y in ~[20.75, 29.25]) gets three choke pillars.
  #  - Pillar (5, 5) is reachable from start via a slow left turn — used by AC6/TC4.
  - shape: {name: 'circle', radius: 0.6}
    state: [ 5,  5, 0]
  - shape: {name: 'circle', radius: 0.6}
    state: [12,  8, 0]
  - shape: {name: 'circle', radius: 0.6}
    state: [35,  5, 0]
  - shape: {name: 'circle', radius: 0.6}
    state: [45,  8, 0]
  - shape: {name: 'circle', radius: 0.6}
    state: [ 8, 25, 0]
  - shape: {name: 'circle', radius: 0.6}
    state: [25, 25, 0]
  - shape: {name: 'circle', radius: 0.6}
    state: [38, 25, 0]
  - shape: {name: 'circle', radius: 0.6}
    state: [10, 35, 0]
  - shape: {name: 'circle', radius: 0.6}
    state: [ 5, 45, 0]
  - shape: {name: 'circle', radius: 0.6}
    state: [30, 42, 0]
  - shape: {name: 'circle', radius: 0.6}
    state: [45, 35, 0]
  - shape: {name: 'circle', radius: 0.6}
    state: [40, 45, 0]
```

## Error Handling

- **Invalid action shape / NaN / Inf:** `step()` raises `ValueError` before touching irsim. Surfaces planner bugs early.
- **`step()` after `done`:** raises `RuntimeError` with message "Episode is done; call reset() first." Caller bug, not recoverable in-place.
- **YAML file missing or unparseable:** `Arena.__init__` lets the underlying `yaml.safe_load` / `irsim.make` exception propagate. No swallow-and-rethrow.
- **YAML start/goal blocked under inflation:** detected by AC11's validation. If T5 reveals this, T1's YAML gets revised — not an `arena.py` runtime concern.
- **YAML lidar beam count mismatch:** `Arena.__init__` raises `ArenaConfigError` (subclass of `ValueError`) if `robot.sensors.lidar2d.number != 360`. Validated once at construction.
- **`numpy.random.SeedSequence(seed)` rejects negative seed:** let it propagate; the harness supplies non-negative seeds.
- **`get_lidar_scan()` returns falsy mid-episode:** Arena returns an all-NaN `(360,)` lidar array and sets `info.lidar_status = "missing"`. Episode continues.
- **`get_lidar_scan()` returns a dict missing the `'ranges'` key:** Arena raises `ArenaRuntimeError`. Indicates an irsim contract violation.

## Testing Strategy

**Levels:** Unit (Arena instantiation, info dict shape, sticky done, reset behavior), Integration (step loop with rendering off, deliberate crash, timeout fire), Layout validation (re-uses `manual_astar.py`'s existing inflation check).

There is no pytest harness in this repo (per project CLAUDE.md). Tests live as **executable assertions inside the smoke-test `__main__`** of `arena/arena.py`, behind a `--check` flag. This keeps the test surface a single command — `python arena/arena.py arena/arena_v1.yaml --check` — and matches the existing repo style (no test runner).

**Convention for the table below:** all `arena.step(...)` calls take a `np.ndarray` of shape `(2, 1)` dtype `float64` as `[[v], [w]]`, per API Contracts. Shorthand like `[[v, w]]` is forbidden.

| ID   | Test Case                                                                                                  | Type        | Expected Behavior                                                  |
|------|------------------------------------------------------------------------------------------------------------|-------------|--------------------------------------------------------------------|
| TC1  | `Arena('arena/arena_v1.yaml', seed=0)` constructs                                                          | Unit        | Returns an `Arena`; no exception                                   |
| TC2  | `arena.reset()` return shapes & info fields                                                                | Unit        | `state.shape == (3,)`, `lidar.shape == (360,)`, `lidar.dtype == float64`, `info` has all 9 fields with correct types, `info.lidar_status == "ok"` |
| TC2b | Missing-lidar tick: monkeypatch `robot.get_lidar_scan` to return `None`                                    | Unit        | `lidar.shape == (360,)`, `np.all(np.isnan(lidar))`, `info.lidar_status == "missing"` |
| TC3  | `arena.step(np.array([[0.0], [0.0]], dtype=float))` once after reset                                       | Unit        | `done == False`, `info.sim_time` increased by `dt`, `info.step_idx == 1` |
| TC4  | Drive forward + slow left turn: repeatedly call `arena.step(np.array([[1.0], [0.3]], dtype=float))`        | Integration | Within 200 steps, `info.crashed == True`, `done == True` (collides with pillar at (5,5)) |
| TC5  | Stand still: repeatedly call `arena.step(np.array([[0.0], [0.0]], dtype=float))` past 120 sim s            | Integration | `info.timed_out == True`, `done == True`                           |
| TC6  | `step()` after `done == True`                                                                              | Unit        | Raises `RuntimeError`                                              |
| TC7  | `reset()` after `done == True` (under whichever path T0 selects)                                           | Unit        | `info.sim_time == 0.0`, `info.step_idx == 0`, all flags `False`    |
| TC8  | Goal reached: monkeypatch `robot.arrive_flag = True` and call `arena.step(np.array([[0.0], [0.0]]))` once  | Unit        | `info.reached_goal == True`, `done == True`                        |
| TC9  | `step()` with action shape `(3,)`, or containing `NaN`, or containing `Inf`                                | Unit        | Raises `ValueError` for each case                                  |
| TC10 | `manual_astar.py`'s validation: `world = load_world(path); grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN); validate_start_and_goal(world, grid)` | Unit        | No exception (start & goal unblocked under inflation)              |
| TC11 | YAML schema fields                                                                                          | Unit        | `world.width == 50`, `world.height == 50`, robot start `[2, 2, 0]`, goal `[48, 48, 0]`, exactly 14 obstacles, 2 are rectangles, 12 are circles |
| TC12 | Construct with a YAML whose `lidar2d.number != 360`                                                         | Unit        | Raises `ArenaConfigError`                                          |

**Test data:** All tests use `arena/arena_v1.yaml` (plus an inline-built tweaked YAML for TC12). irsim is the system under test together with `arena.py`.

**Run command:**

```powershell
.venv\Scripts\Activate.ps1
python arena/arena.py arena/arena_v1.yaml --check     # runs TC1..TC12 headless, prints PASS/FAIL per case
python arena/arena.py arena/arena_v1.yaml --render    # interactive smoke test (AC1)
```

## Tasks

| ID | Task                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Blocked By | Risk | Files                                            | Description |
|----|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|------|--------------------------------------------------|-------------|
| T0 | **Verify irsim API before any class implementation.** Use `mcp__plugin_context7_context7__resolve-library-id` with "ir-sim" / "irsim" and `query-docs` to confirm: (a) whether `env.reset()` exists and restores robot pose + clears internal counters (→ PATH A) or not (→ PATH B); (b) the attribute name for sim step time (`env.step_time`? `env.dt`? other?); (c) whether `irsim.make(yaml, display=...)` (or `disable_all_plot=...`) suppresses the render window; (d) whether `alpha` is a valid `lidar2d` YAML key. Inspect `.venv/Lib/site-packages/irsim/` source for any answer Context7 cannot pin. Record results in a "Verified Assumptions" block appended to this spec file. Satisfies prereq for AC9 / AC1 / AC7. | —          | high | (this spec file — append Verified Assumptions section) | The single source of truth for all "speculative" notes in the spec. Until T0 records a verdict, T3 may not start. If irsim cannot suppress its window for headless mode, flag this immediately — Phase 4 batch sweep depends on it. |
| T1 | Author `arena/arena_v1.yaml` matching the schema in "API Contracts" above. 14 obstacles total (2 walls + 12 pillars). No `_tmp_` prefix. Verify it loads via `python -c "import yaml; yaml.safe_load(open('arena/arena_v1.yaml'))"`. Satisfies AC1, AC11, TC10, TC11.                                                                                                                                                                                                          | —          | med  | `arena/arena_v1.yaml`                            | Write the YAML literally per the spec block. Do not change values during this task — if the layout proves bad in T5, revisit T1 in a follow-up. |
| T2 | Create `arena/__init__.py` (re-exports `Arena`, `EpisodeInfo`) and `arena/arena.py` skeleton: imports, `EpisodeInfo` frozen dataclass, `ArenaConfigError` + `ArenaRuntimeError` exception classes (both subclassing `ValueError` / `RuntimeError` respectively), `Arena` class shell with `__init__` storing config, public method signatures (`reset`, `step`, `close`, `initial_dynamic_snapshot`) raising `NotImplementedError`. Satisfies AC2 partially.                  | —          | low  | `arena/__init__.py`, `arena/arena.py`            | Mirror `manual_astar.py` style: frozen dataclasses, type hints everywhere, `from __future__ import annotations` at top, `UPPER_SNAKE_CASE` module-level constants for `DEFAULT_TIMEOUT_S = 120.0`, `LIDAR_BEAM_COUNT = 360`. No magic numbers in function bodies. No `ArenaConfig` dataclass — pass kwargs directly. |
| T3 | Implement `Arena.__init__` (build `irsim.make(...)` with display flag per T0, derive `traffic_rng` + `motion_rng` via `SeedSequence.spawn(2)`, cache `dt` from env per T0's attribute name, validate `lidar2d.number == 360` else raise `ArenaConfigError`, cache `goal_xy` from `env.robot_list[0].goal[:2, 0]` (NOT a second yaml.safe_load), init `_done=False`, `_step_idx=0`), `reset()` (PATH A: `env.reset()`; PATH B: `env.end()` + `irsim.make(...)` rebuild + re-cache handles, then reset counters and rngs, return initial `(state, lidar, EpisodeInfo)`), `step(action)` (guard `_done`; validate action shape `(2,1)`, finite, float dtype; call `env.step([action])`; flatten `robot.state[:, 0]` → shape `(3,)`; extract lidar per API contract; increment counter; compute `sim_time = step_idx * dt`; read `collision_flag` / `arrive_flag`; check timeout; build `EpisodeInfo`; set sticky `_done`; return), `close()` (call `env.end()`; idempotent via `_closed` flag). Satisfies AC3-AC10, TC1-TC9, TC12. | T0, T2     | high | `arena/arena.py`                                 | Critical implementation details: (1) `robot.state` is shape `(3,1)` — always flatten via `[:, 0]` before returning. (2) `get_lidar_scan()` returns a dict; extract `'ranges'` and shape-check. (3) `wallclock_per_step`: wrap `env.step()` in `time.perf_counter()`. (4) `distance_to_goal`: `float(np.linalg.norm(robot.state[:2, 0] - self._goal_xy))`. (5) After `done` is set, all subsequent `step()` calls raise — guard at top of `step()`. (6) `traffic_seed, motion_seed = ss.spawn(2)` — order matters; document in one-line comment. |
| T4 | Add `__main__`: `argparse` with positional `yaml_path`, `--seed` (default 42), `--render` (store_true), `--check` (store_true). `--check` runs TC1-TC12 sequentially, prints `PASS`/`FAIL <reason>` per case, exits non-zero on any failure. `--render` runs an empty-action loop (`np.array([[0.0], [0.0]], dtype=float)`) at render rate until `done == True`, then prints `done={info}` and calls `close()`. Satisfies AC1, AC12 (smoke command exists).                  | T3         | low  | `arena/arena.py`                                 | TC4 (deliberate crash) and TC5 (timeout) are slow — gate them behind `--check` only, never `--render`. Use `try/finally` around each TC to always call `arena.close()`. TC12 constructs a tweaked YAML on-the-fly (write to a temp path, then dispose). |
| T5 | Manual verification pass: activate venv, run `python arena/arena.py arena/arena_v1.yaml --check` and confirm all TCs PASS. Run `python arena/arena.py arena/arena_v1.yaml --render` and visually verify the window matches the AC1 description (2 walls, 12 pillars, robot at (2,2), goal at (48,48)). If anything fails, file the specific failure back to T1/T3/T4 — do NOT silently tweak constants to make tests pass. Satisfies AC1-AC12 verification.                  | T1, T4     | low  | (observation only; no edits)                     | This is the gate for declaring Phase 0 done. If TC10 (manual_astar.py validation) fails, the YAML pillar positions are too close to start/goal/walls — fix T1, do not relax `SAFETY_MARGIN`. |
| T6 | Append a new section to project `CLAUDE.md` under "The three controllers, at a glance" titled "## The arena harness (Phase 0)". Document: location of `arena/`, the `Arena` class API surface, the smoke + check commands, and the fact that Phase 2 will add dynamic obstacles behind `initial_dynamic_snapshot`. Satisfies AC12.                                                                                                                                              | T5         | low  | `CLAUDE.md`                                      | Match the existing CLAUDE.md tone: terse, code-first. ~10-15 lines, not a tutorial. |

## Notes for Implementer

- **T0 is non-optional.** Three load-bearing irsim claims (reset(), step_time attribute, display kwarg) are unverified in this spec. T0 pins them before T3. If you skip T0, expect AC9 / AC7 / Phase 4 batch sweep to fail in surprising ways.
- **`dt` source:** read from whatever attribute T0 confirms (`env.step_time` is a guess). If irsim does not expose dt, fall back to `0.05` and add a constant `DEFAULT_DT = 0.05`. The `dt` used in `sim_time` accounting must match what irsim actually integrates.
- **Lidar return semantics:** `get_lidar_scan()` returns a dict per `manual_obstacle.py:32`. Extract `'ranges'` and shape-check. Treat falsy return as missing-lidar (NaN-fill). Treat missing `'ranges'` key on a non-falsy scan as a hard error (`ArenaRuntimeError`).
- **`SeedSequence.spawn(2)` order:** `traffic_seed, motion_seed = ss.spawn(2)`. Future Phase 2 spawner consumes `traffic_rng`; future dynamic-obstacle motion consumes `motion_rng`. Document this in a one-line comment in `__init__`.
- **`close()` idempotency:** calling `close()` twice should not raise. Guard with a `_closed` flag.
- **`--check` timing:** TC5 (timeout fire) runs 120 sim seconds of empty steps — at `dt=0.05` that's 2400 steps. Headless this is ~1 s wallclock; rendered it's 2 minutes. Hence `--check` always runs headless.
- **No tests of `initial_dynamic_snapshot`:** it returns `()` in Phase 0. Phase 2 owns its real testing.
- **Forced zig-zag layout:** Walls are 30 m in a 50 m arena, intentionally narrow on one side. Wall A's left gap is `[0, 4.65]` (4.65 m wide after inflation); Wall B's right gap is `[45.35, 50]` (4.65 m). The natural SW→NE path uses Wall B's *left* gap (15 m wide) and Wall A's *right* gap (15 m wide). The narrow gaps exist so Phase 6 / 6b can later test whether a planner ever discovers the "wrong-side" detour under heavy traffic. If T5 reveals A* in `manual_astar.py` fails on this layout, file the failure to T1 — most likely a pillar position needs ~1 m nudging, not a wall change.
- **Rollback:** delete `arena/`, revert the `CLAUDE.md` edit, revert the appended "Verified Assumptions" section in this spec file. Repo is back to start state.
- **What this plan deliberately does not address:** episode metrics writer (Phase 1 owns `results/<algorithm>/<seed>.json`), reproducibility regression test (Phase 1/3), `DynamicObstacle` / `TrafficSpawner` (Phase 2). If the executor finds themselves writing code for any of those, stop — that work belongs in a different plan.

## Verified Assumptions (T0 output)

**Verified:** 2026-05-26 by T0 agent
**irsim version:** 2.9.3 (from `pip show ir-sim`; `.venv/Lib/site-packages/irsim/version.py` reads it dynamically via `importlib.metadata.version("ir-sim")`)

### Q1: env.reset() — PATH A or PATH B?
**Verdict:** PATH A
**Evidence:** `irsim/env/env_base.py:695-723` defines `EnvBase.reset()`. Internal sequence: (1) `self._reset_all()` at line 715 restores objects to YAML state; (2) `self.step(action=[np.zeros((2,1))] * self.robot_number)` at line 716 is a warm-up step that transiently increments `world.count` to 1 and runs `_status_step` + `_objects_sensor_step`; (3) `self._world.reset()` at line 717 — defined at `irsim/world/world.py:239-245` — zeros `world.count` and `world._wp.count` back to 0; (4) `reset_plot()` at line 718 re-inits drawing on the existing matplotlib figure (no new window); (5) `set_status("Reset")` at line 719.
**Post-reset state (verified):** `env._world.count == 0`, `env._world.time == 0.0`. The warm-up step IS undone by `_world.reset()`.
**Implication for T3:**
- `reset()` calls `env.reset()` (no rebuild needed).
- Arena MUST still maintain its own `_step_idx` and `_sim_time` counters — not because irsim's count is non-zero post-reset, but to decouple Arena's contract from any future irsim refactor of its internal counter semantics.
- **Footgun:** the warm-up step's `_status_step` re-evaluates `arrive_flag` / `collision_flag` against the just-reset start pose. Harmless for arena_v1's `(2,2)` start (far from goal and obstacles), but if a future YAML places the start at the goal or in an inflated obstacle, those flags will be `True` immediately after `reset()`. T3 should re-clear them defensively after calling `env.reset()`, or accept that AC9's "all flags False" guarantee depends on the YAML keeping start and goal far apart.

### Q2: dt attribute name
**Verdict:** `env.step_time` (property on `EnvBase` that delegates to `env._world.step_time`)
**Default value if unset:** `0.1` seconds
**Evidence:** `.venv/Lib/site-packages/irsim/env/env_base.py:1252-1260` defines the `step_time` property: `return self._world.step_time`. `.venv/Lib/site-packages/irsim/world/world.py:53` declares `step_time: float = 0.1` as the constructor default and `world.py:105` stores it. The constructor log at env_base.py:224 reads `self._world.step_time` directly, confirming this is the canonical source. Both `env.step_time` and `env._world.step_time` work; `env.step_time` is the public API.
**Implication for T3:** In `Arena.__init__`, cache `self._dt = float(self._env.step_time)` once after `irsim.make(...)`. Do NOT use `env.dt` (no such attribute) or hard-code `0.05` (the `DEFAULT_DT = 0.05` fallback noted in "Notes for Implementer" is unnecessary — `step_time` is guaranteed to exist with a default of 0.1). Compute `sim_time = self._step_idx * self._dt` in `step()`.

### Q3: Headless mode kwarg
**Verdict:** `display=False` (idiomatic; documented in both `irsim.make()` docstring and `EnvBase.__init__` signature)
**Evidence:** `.venv/Lib/site-packages/irsim/__init__.py:78-83` documents `display (bool)` as a `make()` kwarg. `.venv/Lib/site-packages/irsim/env/env_base.py:139` declares `display: bool = True` in `EnvBase.__init__`. `env_base.py:159-160`: when `display=False`, irsim switches matplotlib to the non-interactive `Agg` backend (`matplotlib.use("Agg")`), suppressing the render window. The separate `disable_all_plot=True` kwarg (env_base.py:140, 162) skips plot creation entirely and is documented as more aggressive ("no visualization will be created even if display is True") — but `display=False` is the simpler, idiomatic choice for headless batch sweeps because it still constructs the plot machinery (so any code that calls `env.render()` or reads `_env_plot` does not crash) while suppressing the window.
**Implication for T3 / T4:** In `Arena.__init__`, pass `display=render` to `irsim.make(yaml_path, display=render)` where `render` is the Arena constructor flag (default `False`). Phase 4's batch sweep gets headless via `Arena(..., render=False)`. No need for `disable_all_plot`.

### Q4: alpha key on lidar2d
**Verdict:** VALID (optional plotting opacity)
**Evidence:** `.venv/Lib/site-packages/irsim/world/sensors/lidar2d.py:86` declares `alpha: float = 0.3` as a named kwarg on `Lidar2D.__init__`. Docstring at line 40 ("alpha (float): Transparency for plotting") and line 60 ("alpha (float): Transparency level for plotting the laser beams. Default is 0.3") confirm it is the plot opacity. It is consumed at lines 435 and 447 (`alpha=self.alpha` in the `LineCollection` / `Line3DCollection` constructors). `obstacle_harder.yaml` already sets `alpha: 0.2` and runs cleanly under `manual_obstacle.py` on this irsim version.
**Implication for T1:** Keep `alpha: 0.2` in `arena/arena_v1.yaml`'s `lidar2d` entry. No parser rejection risk on irsim 2.9.3.
