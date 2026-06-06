# Phase 6 (Slice 1) — Planner Interface + Grid Family + D\* Lite Plan

**Goal:** Lay Mission.md Phase 6's foundation — a unified planner interface and a planner-agnostic runner loop — and ship the three *grid-substrate* planners on it: A\* and Dijkstra (each `_once` + `_replan_K`) and incremental D\* Lite. This validates three of the four execution patterns (plan-once, periodic-replan, incremental) with real algorithms, leaving only the reactive pattern (DWA/APF) and sampling planners (RRT/RRT\*) for follow-on plans.

**Approach:** Replace the runner's hard-wired "plan-once → `WaypointFollower` → drive forever" loop with a unified `Controller` protocol (`reset(...)` then a per-step `act(state, lidar) -> action`). A reusable `PathFollowingController` base owns the cached static occupancy grid, a memoryless **lidar→grid fold**, the replan counter, and the `WaypointFollower`, so A\* and Dijkstra are thin `compute_path()` subclasses differing only by heuristic (`h=0` for Dijkstra). The existing tested A\* search (`manual_astar`) is reused unchanged for the static `_once` path; replanning planners fold the live 360-beam lidar onto a copy of the static grid each K steps and re-search from the robot's current cell. D\* Lite is hand-rolled on the *same* grid + fold (referencing an MIT implementation for correctness), updating only the cells the lidar fold changes each step. No new runtime dependency. Ships as two task-groups in one plan: the foundation + A\*/Dijkstra first (independently PR-able), then D\* Lite depending on it.

## Scope

- **In scope:**
  - **`planners/_types.py`** — new `Controller` protocol (`name`, `reset`, `act`). `Path` type retained.
  - **`planners/_grid.py` (new)** — shared grid substrate: `load_lidar_geometry()`, `lidar_to_occupancy()` (the fold), `segment_is_clear_grid()`, `grid_path_to_waypoints()`, the `PathFollowingController` base, and the `build_controller()` factory.
  - **`planners/a_star.py`** — `AStarOnceController` (reuse existing `plan_waypoints`, byte-identical) + `AStarReplanController`.
  - **`planners/dijkstra.py` (new)** — `DijkstraOnceController` + `DijkstraReplanController` (= A\* pipeline with a zero heuristic).
  - **`planners/d_star_lite.py` (new)** — hand-rolled incremental `DStarLiteController` + its `DStarLiteSearch` core (Koenig–Likhachev), on our grid + fold.
  - **`manual_astar.py`** — two small *additive, backward-compatible* changes: `astar_search(..., heuristic_fn=<Euclidean default>)` and `compute_action_from_state(state_xyt, follower)` (with `compute_action(robot, follower)` delegating to it). No behavior change for existing callers/TCs.
  - **`runners/run_episode.py`** — redesign the main loop to be planner-agnostic (`controller.reset()` then `while not done: controller.act(...)`); add `--replan-k N`; build the controller via the registry factory; results subdir uses `algorithm_label(name, replan_k)`. Stop reaching into `arena._robot`.
  - **`runners/run_experiment.py`** — forward `--replan-k` to each child; manifest dir uses the same label.
  - **`arena/arena.py`** — new TC functions for the fold, dispatch, Dijkstra, replan, fallback, label, and D\* Lite (incremental correctness); register them; update the `--check` help/count wording.
  - **Docs** — `CLAUDE.md` "## Phase 6 (grid planners + D\* Lite)" section; `README.md` status row + "Adding a planner" update; one-line `Mission.md` "landed" note.
- **Out of scope (explicit — follow-on plans):**
  - **Reactive planners** DWA, APF (`act` directly returns `(v, ω)`, no follower) — Mission.md Phase 6. The `Controller` interface is *designed* to host them; none ship here.
  - **Sampling planners** RRT, RRT\* (`_once` + `_replan_K`) and the **seeded planner-RNG plumbing** they need — Mission.md Phase 6.
  - **Phase 6b K-sweep automation** (the sweep over `{1, 5, 25, 100, ∞}` and its per-planner K-vs-metrics plot). `--replan-k` is plumbed; the *sweep loop* is not built.
  - **Phase 4 aggregation** (per-algorithm distributions / failure rate) and **Phase 5 `results/plot.py`**.
  - **Privileged ground-truth replanning** (querying the spawner for exact obstacle positions) — rejected as non-Mission-faithful (see Decisions).
  - **Persistent accumulated lidar map** — rejected in favor of memoryless fresh union (see Decisions).
  - **A pytest harness.** Tests stay as `TCi` functions under `arena/arena.py --check`, per project convention.
  - **Changing `Arena` behavior.** Arena is consumed unchanged; arena.py edits are test-only TC additions.

## Decisions

- **Unified `Controller` protocol + shared `PathFollowingController` base** (user-selected) — one `reset(world, snapshot, lidar0, state0)` + `act(state, lidar) -> (2,1) action` shape; the runner becomes a single planner-agnostic `act()` loop. Rejected: keep `plan()` + runner-side dispatch (runner grows branching and reactive planners would force another runner redesign next slice); minimal bolt-on (no clean seam for reactive/incremental).
- **Reuse our A\*; Dijkstra = `h=0`; hand-roll D\* Lite on our grid** (user-selected) — the repo already has a tested, integrated 8-connected/octile/no-corner-cutting A\* wired to `path_to_waypoints` + `WaypointFollower` + TC14/15; replacing it with a library is a net loss. Dijkstra is that search with a zero heuristic. No maintained, pip-stable, *incremental*, grid-pluggable D\* Lite exists (the candidates — `python-motion-planning`, PythonRobotics, Sollimann/mdeyo — own their own map and would force an adapter + diverge the substrate from our A\*, muddying the Phase 7 comparison; one-shot library "D\* Lite" also discards the very incrementality Phase 7 measures). So D\* Lite is hand-rolled in `manual_astar` style on our grid + fold, using an MIT implementation as a *correctness reference only*. No new runtime dependency.
- **Lidar-only after t=0 (Mission-faithful)** (user-selected) — replanners fold ONLY the live 360-beam lidar into the grid; no privileged access to true obstacle positions. The seam in `arena.py`'s `initial_dynamic_snapshot` docstring ("query the spawner separately") is deliberately NOT used.
- **Memoryless fresh union fold** (user-selected) — each replan starts from a cached copy of the precomputed static grid and adds blocked cells from the *current* lidar frame only; prior frames are discarded (no stale "ghost" obstacles).
- **Keep last valid path on mid-episode replan failure** (user-selected) — if a replan finds no path (transient traffic box-in, start/goal cell momentarily blocked), the failed replan is ignored and the controller keeps following the most recent good path. The **t=0** plan failing still raises → `planner_error` (unchanged TC16 behavior). This distinction is load-bearing: `reset()` failures propagate; `act()` replan failures are swallowed.
- **Keep `_once` entries separate; new `_replan` family entries** (user-selected) — `a_star_once` stays its own registry entry (already shipped + TC-covered); `a_star_replan` / `dijkstra_replan` are new entries taking `--replan-k`. `d_star_lite` is its own entry with no `_once`/`_replan` split (Mission.md: incremental, updates only affected cells). Rejected: unify `_once` as `replan K=∞` (redefines the shipped `a_star_once` identity).
- **`--replan-k N` flag; results subdir label `<family>_k<K>`** (user-confirmed) — `--replan-k` is REQUIRED for `_replan` families, REJECTED for `_once` and `d_star_lite`. The episode output dir is `results/<world_stem>/<algorithm_label>/` where `algorithm_label = name` for non-replan and `f"{name}_k{replan_k}"` for replan (so the future 6b sweep over K never clobbers). Forwarded through `run_experiment`.
- **Determinism relaxed for the new planners** (user-selected) — no new byte-identity AC/TC is asserted for replan/D\* Lite traces. (In practice they remain byte-deterministic — irsim lidar `noise=False`, traffic seeded — but this phase does not gate on it.) The existing Arena/traffic determinism (TC20/TC24) is untouched. Byte-identity is still asserted where it is free: the deterministic static `_once` planners.
- **`a_star_once` re-baseline allowed but not expected** (user-selected) — the redesigned plan-once path reuses `plan_waypoints` + identical action math, so `a_star_once` trace/metrics bytes should be unchanged and TC14/TC15/TC24 keep passing as-is. If an unforeseen byte diff appears, those TCs are re-baselined rather than the design contorted to preserve old bytes.
- **Two waypoint-extraction paths, kept separate** — the static `_once` planners keep the existing analytic-LOS `path_to_waypoints` (over `ObstacleSpec`s) so `a_star_once` stays byte-identical. Replanners + D\* Lite use a NEW *grid-native* `grid_path_to_waypoints` (stride sampling + grid-cell line-of-sight bisection) because lidar hits live in the folded grid, not in the analytic `ObstacleSpec` list. This is NOT redundant duplication: `path_to_waypoints` is pinned independently by the standalone `manual_astar.py` demo (`run_simulation` → `plan_waypoints` → `path_to_waypoints`), which this plan does not touch, so it exists regardless; `grid_path_to_waypoints` is independently mandatory for replanning. Routing `a_star_once` through the grid extractor would delete zero code while forcing a TC14/15/24 re-baseline — pure cost, rejected.
- **`_once` controllers are standalone; only the `_replan` / D\* Lite family extends `PathFollowingController`** — `AStarOnceController` / `DijkstraOnceController` are thin standalone `Controller`s whose `reset()` calls `plan_waypoints` (analytic, re-derives start/goal from the YAML, ignores `state0`/`lidar0`) and whose `act()` is `compute_action_from_state`. They do NOT subclass `PathFollowingController` (whose `reset()` builds the initial path via the grid `compute_path`). Conflating the two would route `_once` through the grid extractor and break byte-parity.
- **`initial_snapshot` is accepted but intentionally unused by the grid planners** — Mission.md grants the planner the t=0 obstacle positions, but `lidar0` (sensed after the obstacles are spawned, per `arena.reset()` step 4) already encodes those same obstacles, and replanners are lidar-only by decision. So `Controller.reset` takes `initial_snapshot` for interface generality (reactive/sampling planners may use it later) while A\*/Dijkstra/D\* Lite ignore it and plan on the static grid + `lidar0` fold. This is a deliberate divergence, not a dropped input.
- **`ALGORITHMS` is populated in two stages across the two-group split** — four grid keys in Group A (T3), the fifth `d_star_lite` key in Group B (T11). The runner's argparse `choices=list(ALGORITHMS)` reflects whichever keys are registered when a PR lands, so `--algorithm d_star_lite` is intentionally an invalid CLI choice until Group B. Group A tests must not freeze the registry length or assert exact-set equality.
- **Ships as two task-groups, one plan** (user-selected) — Group A (interface + runner + fold + A\*/Dijkstra) is independently PR-able and proves the interface with simple planners; Group B (D\* Lite) depends on Group A. De-risks the hardest algorithm by landing the substrate first.

## Acceptance Criteria

- [ ] **AC1 (interface):** `planners/_types.py` exposes a `@runtime_checkable Controller` Protocol with `name: str`, `reset(world_yaml, initial_snapshot, lidar0, state0) -> None`, and `act(state, lidar) -> np.ndarray` (shape `(2,1)`, float). All five shipped controllers satisfy it.
- [ ] **AC2 (runner is planner-agnostic):** `runners/run_episode.py`'s main loop contains no planner-type branching and no `arena._robot` access — it calls `controller.reset(...)` once (t=0 plan; `ValueError`/`RuntimeError` → `planner_error`, trace deleted, exit 0) then `while not done: action = controller.act(state, lidar); arena.step(action)`. The step-0 trace anchor is still written before `reset()`.
- [ ] **AC3 (`a_star_once` preserved):** `a_star_once --no-traffic` reaches the goal and produces a trace whose 7-key schema is unchanged; two same-seed runs are byte-identical (TC15-style). TC14/TC15/TC16/TC24 still PASS (re-baselined only if an unforeseen diff appears, noted in the PR).
- [ ] **AC4 (lidar fold):** `lidar_to_occupancy(static_cells, grid, state, lidar, geom, inflation)` returns a NEW boolean grid = static ∪ {cells within `inflation` of each non-NaN lidar return mapped to world frame at the current pose}; NaN beams are skipped; bearings are `np.linspace(angle_min, angle_max, number)` from the YAML sensor block; the static grid argument is not mutated. Verified by TC.
- [ ] **AC5 (Dijkstra):** `dijkstra_once --no-traffic` reaches the goal; on the static grid its path cost equals A\*'s (both optimal). Implemented as the shared search with a zero heuristic — no duplicated search loop. The grid-extractor waypoints (`grid_path_to_waypoints`) it would use on a *replan* must also pass an inflation-aware clearance check (no waypoint segment clips an inflated obstacle), not only match A\*'s grid cost.
- [ ] **AC6 (replan family + label):** `python -m runners.run_episode --algorithm a_star_replan --replan-k 5 --world arena/arena_v1.yaml --seed S` writes to `results/arena_v1/a_star_replan_k5/`, runs to completion, and (traffic on) emits 8-key trace lines. `dijkstra_replan` behaves identically. `--replan-k` is required for `_replan` families and rejected (argparse/validation error, exit 2) for `_once` and `d_star_lite`.
- [ ] **AC7 (replan cadence + memoryless fold):** an `a_star_replan` controller recomputes its path on exactly every `K`-th `act()` call (counter-driven; `compute_path` invocation count == `floor(act_calls / K)`), and each replan's occupancy is static ∪ current-frame fold only (no accumulation across frames). Verified by TC counting `compute_path` invocations against the `act()` call index.
- [ ] **AC8 (mid-replan failure fallback):** when a mid-episode `compute_path()` raises (e.g. injected fold boxes the robot in), `act()` swallows it and returns an action from the previously valid follower — no exception escapes, episode continues. The t=0 `reset()` plan failing still raises → `planner_error`.
- [ ] **AC9 (D\* Lite reaches goal):** `d_star_lite --no-traffic` reaches the goal; on the static map its initial path cost equals A\*/Dijkstra's (optimal) AND its grid-extractor waypoints pass the same inflation-aware clearance check as AC5. Trace schema intact. End-to-end with traffic runs to completion.
- [ ] **AC10 (D\* Lite incremental correctness):** after a map change is applied to the D\* Lite search, the path it recomputes incrementally equals a fresh from-scratch search on the updated grid. The test must make the change *binding*: the blocked cell must lie on the pre-block optimal path AND the block must strictly increase the optimal cost, so correctness is verified as **post-update cost == fresh-search cost AND post-update cost > pre-block cost** (a no-op update fails). Equality is asserted on cost (and cells modulo equal-cost tie-breaking), never on an exact cell set. Verified by TC36.
- [ ] **AC11 (`run_experiment` passthrough):** `run_experiment --algorithm a_star_replan --replan-k 5 ...` forwards `--replan-k 5` to every child and writes the manifest + episodes under `results/<world_stem>/a_star_replan_k5/`.
- [ ] **AC12 (`--check`):** `python arena/arena.py arena/arena_v1.yaml --check` runs the existing suite plus the new Phase 6 TCs, all PASS, exit 0; the help/docstring TC-count wording is updated to the new total. No "under 120 s" claim reintroduced.
- [ ] **AC13 (no Arena behavior change):** `Arena` and its public API are unchanged; TC1–TC27 still PASS. `manual_astar.py`'s existing public behavior (the demo, `plan_waypoints`, `compute_action(robot, follower)`) is unchanged — the two additions are backward-compatible (default heuristic, delegating wrapper).
- [ ] **AC14 (docs):** `CLAUDE.md` gains a Phase 6 section (interface, fold, the five planner names, `--replan-k`/label rule, replan-failure policy, D\* Lite incrementality, new TCs); `README.md` flips the Phase 6 status and updates "Adding a planner" to the `Controller` shape; `Mission.md` gets a one-line "grid planners + D\* Lite slice landed; reactive/sampling/6b deferred" note.
- [ ] **AC15 (name == registry key):** for every family key `k` in `ALGORITHMS`, `build_controller(k, <valid replan_k>).name == k`, so the results-dir label and manifest dir cannot silently diverge from the requested algorithm. Verified by TC33.

## Data Model

```python
# planners/_types.py
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np
from arena.dynamic import DynamicObstacleState

Path = tuple[np.ndarray, ...]  # ordered (2,)-float64 world-frame waypoints; last == goal

@runtime_checkable
class Controller(Protocol):
    name: str  # e.g. "a_star_replan" — the FAMILY name; results label adds _k<K>

    def reset(
        self,
        world_yaml: str,
        initial_snapshot: tuple[DynamicObstacleState, ...],  # t=0 view; () when traffic off
        lidar0: np.ndarray,                                  # (360,) float64; NaN = no return
        state0: np.ndarray,                                  # (3,) float64 [x, y, theta]
    ) -> None:
        """Build static substrate + t=0 plan. May raise ValueError/RuntimeError
        (no path) -> runner records planner_error."""

    def act(self, state: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        """Return the next action, shape (2,1) float [[v],[w]]. Must not raise on a
        mid-episode replan failure (keep the last valid path)."""
```

```python
# planners/_grid.py  (shared substrate)

@dataclass(frozen=True)
class LidarGeometry:
    angle_min: float        # = -WrapTo2Pi(angle_range)/2
    angle_max: float        # = +WrapTo2Pi(angle_range)/2
    number: int             # beam count (360)

def load_lidar_geometry(world_yaml: str) -> LidarGeometry:
    """Read robot.sensors[lidar2d].{angle_range, number} from the YAML and apply
    irsim's WrapTo2Pi to angle_range (matches Lidar2D.__init__). Bearings are then
    np.linspace(angle_min, angle_max, number) — NOT angle_range/number spacing."""

def lidar_to_occupancy(
    static_cells: np.ndarray,     # (rows, cols) bool — the cached static grid (NOT mutated)
    grid: OccupancyGrid,          # for resolution/offset + world<->grid maps
    state: np.ndarray,            # (3,) [x, y, theta] current pose (lidar offset is [0,0,0])
    lidar: np.ndarray,            # (360,) float64, NaN = no return
    geom: LidarGeometry,
    inflation: float,             # robot_radius + SAFETY_MARGIN
) -> np.ndarray:
    """Return a NEW bool grid = static_cells | (cells within `inflation` of each
    non-NaN return). For beam i with finite range r:
        bearing = geom.angle_min + i*(geom.angle_max-geom.angle_min)/(number-1)
        world_pt = state[:2] + r*[cos(theta+bearing), sin(theta+bearing)]
        mark all cells whose center is within `inflation` of world_pt."""

def segment_is_clear_grid(grid_cells: np.ndarray, grid: OccupancyGrid,
                          p0: np.ndarray, p1: np.ndarray) -> bool:
    """Grid-native line-of-sight: sample along p0->p1 at ~half-resolution; clear iff
    no sampled cell is occupied. (Replanner analogue of segment_is_clear, which uses
    analytic ObstacleSpecs the fold has no access to.)"""

def grid_path_to_waypoints(cells_path: list[tuple[int,int]], grid: OccupancyGrid,
                           grid_cells: np.ndarray, start_xy: np.ndarray,
                           goal_xy: np.ndarray, stride: int) -> Path:
    """Downsample a grid cell path to world waypoints (stride + grid-LOS bisection),
    anchored at the ACTUAL start_xy/goal_xy (not world.start) so replanning from the
    robot's current pose works. Last waypoint == goal_xy."""

class PathFollowingController:
    """Base for grid path-followers. Owns: cached WorldModel + static grid + goal cell,
    LidarGeometry, replan_k counter, current WaypointFollower. Subclasses implement
    compute_path(state, lidar) -> Path."""
    def __init__(self, name: str, heuristic_fn, replan_k: int | None) -> None: ...
    def reset(self, world_yaml, snapshot, lidar0, state0) -> None:
        # load_world, build_occupancy_grid (cache static_cells), validate goal,
        # initial path via compute_path(state0, lidar0), build follower. Raises on no path.
    def act(self, state, lidar) -> np.ndarray:
        # self._k += 1
        # if replan_k is not None and self._k % replan_k == 0:
        #     try: self._follower = WaypointFollower(list(self.compute_path(state, lidar)), ...)
        #     except (ValueError, RuntimeError): pass   # keep last valid path
        # return compute_action_from_state(state, self._follower)
    def compute_path(self, state, lidar) -> Path: raise NotImplementedError
    # Replan compute_path flow (A*/Dijkstra _replan):
    #   folded_cells = lidar_to_occupancy(static_cells, grid, state, lidar, geom, inflation)  # np.ndarray
    #   folded_grid  = OccupancyGrid(cells=folded_cells, resolution=grid.resolution, offset=grid.offset)
    #   cur_cell = world_to_grid(state[:2], grid); cells_path = astar_search(folded_grid, cur_cell, goal_cell, heuristic_fn)
    #   return grid_path_to_waypoints(cells_path, grid, folded_cells, state[:2], goal_xy, WAYPOINT_STRIDE)
    # NOTE: astar_search reads grid.cells, so the folded ndarray MUST be re-wrapped in an OccupancyGrid first.

# Standalone _once controllers (NOT PathFollowingController subclasses):
#   AStarOnceController/DijkstraOnceController.reset(world,...) -> plan_waypoints(world)         [analytic LOS]
#     (DijkstraOnce uses a heuristic_fn=0 variant of the static pipeline; AStarOnce reuses plan_waypoints as-is)
#   .act(state, lidar) -> compute_action_from_state(state, self._follower)

# Registry / factory
ALGORITHMS: dict[str, type[Controller]] = {
    "a_star_once":     AStarOnceController,      # replan_k must be None
    "a_star_replan":   AStarReplanController,    # replan_k required
    "dijkstra_once":   DijkstraOnceController,
    "dijkstra_replan": DijkstraReplanController,
    "d_star_lite":     DStarLiteController,      # incremental; replan_k must be None
}
REPLAN_FAMILIES = frozenset({"a_star_replan", "dijkstra_replan"})

def algorithm_label(name: str, replan_k: int | None) -> str:
    return f"{name}_k{replan_k}" if name in REPLAN_FAMILIES else name

def build_controller(name: str, replan_k: int | None) -> Controller:
    """Validate replan_k against the family, then construct. Raises ValueError if a
    _replan family is missing --replan-k, or a non-replan family was given one."""
```

```python
# planners/d_star_lite.py  (incremental core — Koenig & Likhachev)
class DStarLiteSearch:
    """g/rhs maps, priority queue U keyed (k1,k2), key modifier k_m, s_start, s_goal,
    8-connected octile cost on a bool grid with no-corner-cutting (mirrors astar_search
    connectivity). API: __init__(grid_cells, grid, start_cell, goal_cell);
    compute_shortest_path(); update_cells(changed: list[tuple[int,int]]) (UpdateVertex +
    k_m bump); move_start(new_start_cell); extract_path() -> list[cell]."""

class DStarLiteController:
    """reset(): build static grid + DStarLiteSearch; compute_shortest_path; extract +
    grid_path_to_waypoints -> follower. act(): fold lidar -> new occupancy; diff vs prior
    occupancy -> changed cells; search.update_cells(changed); search.move_start(cur_cell);
    compute_shortest_path; re-extract path -> follower (keep last on failure);
    compute_action_from_state."""
```

```python
# manual_astar.py  (additive, backward-compatible)
def astar_search(grid, start, goal, heuristic_fn=None) -> list[tuple[int,int]]:
    # heuristic_fn defaults to the current Euclidean heuristic(cell, goal); Dijkstra
    # passes lambda *_: 0.0. Existing callers (plan_waypoints, TCs) unchanged.
    # MUST replace ALL THREE internal heuristic() call sites: the initial heap push
    # priority AND each per-neighbor priority. A default that leaves any site on the
    # hardcoded heuristic() makes the param inert for Dijkstra.

def compute_action_from_state(state_xyt: np.ndarray, follower: WaypointFollower) -> np.ndarray:
    # the body of the current compute_action, reading x,y,theta from a (3,) array.
def compute_action(robot, follower):  # now delegates:
    return compute_action_from_state(robot.state[:, 0], follower)
```

## API Contracts

```
python -m runners.run_episode
    --algorithm {a_star_once,a_star_replan,dijkstra_once,dijkstra_replan,d_star_lite}
    --seed <int>            required
    --world <yaml>          required
    [--replan-k <int>]      required iff algorithm in {a_star_replan, dijkstra_replan};
                            rejected (exit 2) for _once and d_star_lite; >= 1
    [--render] [--results-dir DIR] [--traffic|--no-traffic]   (unchanged)

Output dir: <results-dir>/<world_stem>/<algorithm_label>/   where
    algorithm_label = f"{algorithm}_k{replan_k}" if algorithm in REPLAN_FAMILIES else algorithm
Files: <seed>.json (7-key metrics, unchanged) + <seed>.trace.jsonl (7 keys --no-traffic,
       8 keys with traffic, unchanged schema). planner_error path unchanged.
Exit: 0 episode ran (incl. crash/timeout/planner_error) | 2 argparse / Arena config error.

python -m runners.run_experiment
    ... (unchanged) [--replan-k <int>]   # forwarded verbatim to each run_episode child;
                                         # manifest + episodes land under <algorithm_label>/
```

## Error Handling

- **t=0 plan failure (`reset()` raises `ValueError`/`RuntimeError`)** — runner catches, deletes the trace, writes metrics with `planner_error` set, exits 0 (unchanged TC16 behavior).
- **Mid-episode replan failure (`compute_path()` raises inside `act()`)** — swallowed by `act()`; the previous follower is kept; the episode continues. Never surfaces as `planner_error`. (AC8.)
- **`--replan-k` misuse** — missing for a `_replan` family, or supplied for `_once`/`d_star_lite` → validation error before any episode runs (exit 2).
- **Empty initial path** — `reset()` raises `ValueError("planner produced an empty path")` → `planner_error` path (mirrors the current empty-waypoint guard).
- **Lidar all-NaN at a replan** (`lidar_status == "missing"`) — the fold adds nothing; the replan searches the static grid only (degrades to the static plan; never crashes).
- **Robot/goal cell blocked by the fold** — the search raises `RuntimeError` (no path) → caught by the mid-replan fallback (keep last path). At t=0 only, it propagates as `planner_error`.
- **D\* Lite changed-cell diff** — bounded to cells the fold flips this tick (xor of new vs prior occupancy); if the start or goal cell becomes blocked, the search yields no path → fallback keeps the last path.
- **Non-`a_star_v1` world without a sensor block** — `load_lidar_geometry` raises `ArenaConfigError`-style `ValueError` at `reset()` (t=0) → `planner_error` (only reachable on malformed worlds; arena_v1/v2 have the block).

## Testing Strategy

**Levels:** Unit (fold, fallback, Dijkstra==A\*, D\* Lite incremental==from-scratch, label/validation), Integration (each planner end-to-end via subprocess on fast/relevant worlds), Regression (TC1–TC27 unchanged). All as `TCi` functions in `arena/arena.py --check`; no pytest. New TCs prefer `--no-traffic` static worlds (fast, deterministic) or short windows; full traffic drives are kept to one or two smoke TCs.

| ID | Test Case | Type | Expected Behavior |
|----|-----------|------|-------------------|
| TC28 | Lidar→grid fold geometry | Unit | Synthetic pose + a lidar with one finite return at a known beam index → `lidar_to_occupancy` marks the expected world cell(s) blocked, leaves a far cell free, skips NaN beams, and does NOT mutate the static grid argument. Bearings reconstructed via `linspace` match the hit location. **Fold the SAME lidar at two distinct poses (different x,y,θ) and assert the marked cell moves accordingly** — the pose-dependent transform is the most bug-prone part. Pure, instant. |
| TC29 | Dijkstra == A\* optimal + reaches goal | Unit+Integration | On `arena_v1` static grid, `dijkstra_once` path cost == `a_star_once` path cost (both optimal). Subprocess `dijkstra_once --no-traffic` reaches goal; trace schema valid. |
| TC30 | a_star_replan end-to-end + label | Integration | Subprocess `a_star_replan --replan-k 5 --world arena_v1 --traffic` exits 0, terminates, writes to `…/a_star_replan_k5/`, every trace line has 8 keys. Short-ish (A\* crashes/finishes within timeout). |
| TC31 | Replan cadence + memoryless fold | Unit | Drive an `a_star_replan(K=3)` controller with stub state/lidar; count `compute_path` invocations against the `act()` call index and assert it fires on the 3rd, 6th, 9th `act()` call ONLY (not sim-step, not act 0); assert each call's occupancy == static ∪ current frame (no accumulation across frames). |
| TC32 | Mid-replan failure fallback | Unit | Seed the controller with a valid path; feed an `act()` whose injected lidar fold boxes the robot in (search raises) → no exception escapes, returned action matches the prior follower, **and the follower object is unchanged (`controller._follower is <the pre-failure follower>`)** so a partial rebuild can't corrupt the waypoint index. Then a clean frame replans normally. |
| TC33 | `--replan-k` validation + dispatch + name parity | Unit | `build_controller("a_star_replan", None)` raises; `build_controller("a_star_once", 5)` raises; (Group B) `build_controller("d_star_lite", 5)` raises; valid combos construct a `Controller`; **for every key `k` the controllers exercised, `build_controller(k, …).name == k`**. `algorithm_label` returns `a_star_replan_k5` / `a_star_once`. Asserts membership of the keys tested, NOT `len(ALGORITHMS)`. |
| TC34 | a_star_once parity through new loop | Integration | `a_star_once --no-traffic` two same-seed subprocess runs → byte-identical trace JSONL + reaches goal (re-baseline of TC14/15 if bytes shifted, noted). Guards the runner redesign against regressing the shipped path. |
| TC35 | D\* Lite reaches goal + optimal initial | Integration | Subprocess `d_star_lite --no-traffic` reaches goal; initial path cost == A\* cost on the static map; trace schema valid. |
| TC36 | D\* Lite incremental == from-scratch (binding) | Unit | Build `DStarLiteSearch` on a small grid whose unique optimal path is known. (a) `compute_shortest_path` and assert the optimal path **traverses cell C** (pre-condition: C is on the current optimal path). (b) Pick the grid so blocking C **strictly lengthens** the optimum (a strictly costlier detour exists). (c) `update_cells` to block C; `compute_shortest_path` again; assert **post-update cost == a fresh from-scratch A\*/Dijkstra search on the updated grid AND post-update cost > pre-block cost** (the block must bind, so a no-op/ignored update fails). Compare at the cost level only (not exact cell set), per AC10's tie-break allowance. Build the from-scratch oracle from the same `astar_search` so both share the cost model. |
| TC37 | D\* Lite registered + end-to-end with traffic + rejects K | Integration | Assert `"d_star_lite" in ALGORITHMS` and `run_episode --algorithm d_star_lite` parses (the staged Group-B registration actually completed); `d_star_lite --traffic` runs to completion (8-key trace); `--algorithm d_star_lite --replan-k 5` exits 2 (rejected). |

**Test data:** `arena/arena_v1.yaml` (static + traffic), `arena/arena_no_path.yaml` (t=0 failure path is already TC16; reused implicitly). TC28/TC31/TC32/TC36 build small synthetic grids / stub lidar in-process — no new fixture files. The D\* Lite incremental test uses a hand-built occupancy array so the expected from-scratch path is cheap to compute.

**Run commands:**
```powershell
.venv\Scripts\Activate.ps1
python arena/arena.py arena/arena_v1.yaml --check          # existing + TC28–TC37, all PASS

# Group A planners:
python -m runners.run_episode --algorithm dijkstra_once   --seed 42 --world arena/arena_v1.yaml --no-traffic
python -m runners.run_episode --algorithm a_star_replan   --replan-k 5  --seed 42 --world arena/arena_v1.yaml
# Group B:
python -m runners.run_episode --algorithm d_star_lite     --seed 42 --world arena/arena_v1.yaml
# Batch (passthrough):
python -m runners.run_experiment --algorithm a_star_replan --replan-k 5 --world arena/arena_v1.yaml --num-seeds 3
```

## Tasks

### Group A — Interface + runner + fold + A\*/Dijkstra (independently PR-able)

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T0 | Create feature branch from the branch that ACTUALLY contains Phase 3 | — | low | (git) | This plan depends on the Phase 1–3 runner/arena code. The repo is currently on `phase3-reproducibility` (Phase 3 may not be merged to `origin/main` yet). **Branch from the integration ref that contains `runners/run_experiment.py` + the Phase 3 TCs** — i.e. `phase3-reproducibility` (or its merge target once merged), NOT blindly `origin/main`, or you lose the Phase 3 code this slice builds on. Verify with `git log --oneline -5` that the base has the Phase 3 commits before branching. Never commit to `main`. |
| T1 | `manual_astar.py` additive helpers | T0 | med | `manual_astar.py` | Add `heuristic_fn` param to `astar_search` defaulting to the current Euclidean `heuristic`; **rewire ALL THREE internal `heuristic()` call sites** (initial heap push + each per-neighbor priority) to call `heuristic_fn` so the default preserves current bytes and `heuristic_fn=lambda *_: 0.0` actually yields Dijkstra (existing callers unchanged). Add `compute_action_from_state(state_xyt, follower)` holding the body of `compute_action`; make `compute_action(robot, follower)` delegate via `robot.state[:, 0]`. No behavior change for the demo or TCs. Satisfies AC5 (search param), AC13. |
| T2 | `Controller` protocol in `planners/_types.py` | T0 | low | `planners/_types.py`, `planners/__init__.py` | Add the `@runtime_checkable Controller` Protocol (see Data Model). Keep `Path`. **Remove the obsolete `PathPlanner` Protocol** (nothing else imports it) and update `planners/__init__.py` `__all__` to export `Controller`/`Path` (drop `PathPlanner`). Satisfies AC1. |
| T3 | Shared grid substrate `planners/_grid.py` | T1, T2 | high | `planners/_grid.py` (new) | Implement `LidarGeometry` + `load_lidar_geometry` (read YAML sensor block, irsim `WrapTo2Pi`, `linspace` bearings); `lidar_to_occupancy` (memoryless fresh union, returns a NEW `np.ndarray`, skips NaN, current-pose transform); `segment_is_clear_grid`; `grid_path_to_waypoints` (anchored at actual start/goal); `PathFollowingController` base (cache static grid, replan counter, keep-last-on-failure, re-wraps folded cells into `OccupancyGrid` before `astar_search`, `compute_action_from_state`); `REPLAN_FAMILIES`/`algorithm_label`/`build_controller`; and `ALGORITHMS` populated with **EXACTLY the four grid keys** (`a_star_once`, `a_star_replan`, `dijkstra_once`, `dijkstra_replan`). The fifth key `d_star_lite` is added by Group B's T11 — do NOT register it here, so Group A's CLI lists only the four. `build_controller(k).name` must equal `k`. Satisfies AC4, AC7, AC8, AC6 (label/validation), AC2 (base), AC15. |
| T4 | A\* controllers `planners/a_star.py` | T3 | med | `planners/a_star.py` | `AStarOnceController` — **standalone** `Controller` (NOT a `PathFollowingController` subclass): name `a_star_once`; `reset` calls existing `plan_waypoints` for byte-identity (ignores `state0`/`lidar0`); `act` = `compute_action_from_state`. `AStarReplanController` — subclass `PathFollowingController`, Euclidean heuristic, `compute_path` = fold → re-wrap folded cells in `OccupancyGrid(cells=folded, resolution=grid.resolution, offset=grid.offset)` → `astar_search` from current cell → `grid_path_to_waypoints`. **Remove the obsolete `AStarOncePlanner.plan()` adapter.** Satisfies AC3, AC6, AC7. |
| T5 | Dijkstra controllers `planners/dijkstra.py` | T3 | low | `planners/dijkstra.py` (new) | `DijkstraOnceController` + `DijkstraReplanController` = the A\* controllers with `heuristic_fn = lambda *_: 0.0` (no duplicated search). Satisfies AC5. |
| T6 | Runner main-loop redesign | T3, T4, T5 | high | `runners/run_episode.py` | Replace the plan-once loop with: build controller via `build_controller(args.algorithm, args.replan_k)`; `out_dir` from `algorithm_label`; after `arena.reset()` write step-0 anchor, then `try: controller.reset(world, snapshot, lidar0, state0) except (ValueError, RuntimeError): planner_error path`; `while not done: action = controller.act(state, lidar); state,lidar,done,info = arena.step(action); write trace`. Add `--replan-k` (validated in `build_controller`; a `build_controller` `ValueError` is a config error → exit 2, distinct from a `reset()` `ValueError` → `planner_error`, exit 0). Remove `arena._robot`/`compute_action` import; the next `act()` consumes the `state,lidar` threaded back from `arena.step` (never re-read `arena._robot`). Keep metrics/trace writing + fsync intact. **Prove `a_star_once` parity (TC34) the moment the loop + T4 land, before wiring replan families** — that isolates any byte regression to the loop redesign itself. Satisfies AC2, AC3, AC6. |
| T7 | `run_experiment` `--replan-k` passthrough | T6 | low | `runners/run_experiment.py` | Add `--replan-k` arg; forward to each `run_episode` child verbatim; compute `out_dir`/manifest dir via `algorithm_label`. Sweep automation NOT added. Satisfies AC11. |
| T8 | Group A TCs in `arena/arena.py` | T6, T7 | med | `arena/arena.py` | Add TC28, TC29, TC30, TC31, TC32, TC33, TC34; register in `_run_checks`; mirror existing subprocess-TC scaffolding (TC14/TC15/TC22). **TCs MUST assert membership + per-key family/K behavior only for the keys they exercise — never `len(ALGORITHMS)` or exact-set equality** (Group B adds `d_star_lite`). **Update every count-bearing string** for the new total: the `--check` argparse help, the `arena.py` module/`_run_checks` docstring, and `CLAUDE.md`'s "28 PASS"/"TC1–TC27" wording. Verify the new PASS-line count empirically (don't hardcode a guessed number). Satisfies AC4, AC5, AC6, AC7, AC8, AC3, AC12, AC15. |
| T9 | Group A docs | T8 | low | `CLAUDE.md`, `README.md`, `Mission.md` | Add the Phase 6 grid section to `CLAUDE.md` (interface, fold, names, `--replan-k`/label, replan-failure policy, new TCs); flip `README.md` status + update "Adding a planner" to the `Controller` shape; one-line `Mission.md` note worded as "**grid planners + D\* Lite slice landed; reactive (DWA/APF) + sampling (RRT/RRT\*) + 6b sweep still deferred**" (do NOT imply all of Phase 6 shipped). (D\* Lite paragraph added in T13.) **These three doc files are edited again in T13** — if Group A and B ship as two PRs, rebase T13's edits on T9's. Satisfies AC14 (partial). |

### Group B — D\* Lite (incremental), depends on Group A

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T10 | D\* Lite search core | T3 | high | `planners/d_star_lite.py` (new) | Hand-roll `DStarLiteSearch` (Koenig–Likhachev): `g`/`rhs`, priority queue `U` with `(k1,k2)` keys, key modifier `k_m`, 8-connected octile cost matching `astar_search` connectivity + no-corner-cutting, `compute_shortest_path`, `update_cells(changed)` (UpdateVertex over changed cells + their neighbors), `move_start`, `extract_path`. Use an MIT impl as a correctness reference only; code in our style. Satisfies AC10 (core). |
| T11 | D\* Lite controller | T10, T4, T3 | high | `planners/d_star_lite.py`, `planners/_grid.py` | `DStarLiteController` (name `d_star_lite`): `reset` builds static grid + search + initial path via `grid_path_to_waypoints`; `act` folds lidar → new occupancy, diffs vs prior occupancy (xor) → changed cells, `update_cells` + `move_start(cur_cell)` + `compute_shortest_path`, re-extract path → follower (keep last on failure), `compute_action_from_state`. **Add the fifth `d_star_lite` entry to the existing `ALGORITHMS` dict in `planners/_grid.py`** (do not redefine/reorder the four Group-A keys); this is where `--algorithm d_star_lite` becomes a valid CLI choice. `build_controller` rejects `--replan-k` for it, and `build_controller("d_star_lite").name == "d_star_lite"`. Blocked-By T3 because it edits `_grid.py`. Satisfies AC9, AC2, AC15. |
| T12 | Group B TCs | T11, T8 | med | `arena/arena.py` | Add TC35 (reaches goal + optimal initial), TC36 (incremental == from-scratch on a synthetic grid), TC37 (traffic end-to-end + rejects `--replan-k`); register; update count wording. Satisfies AC9, AC10, AC12. |
| T13 | Group B docs | T12 | low | `CLAUDE.md`, `README.md`, `Mission.md` | Add the D\* Lite paragraph to the `CLAUDE.md` Phase 6 section (incremental update from the fold, no `_once`/`_replan` split, rejects `--replan-k`, TC35–TC37); update the `README.md` planner list; extend the `Mission.md` note. Satisfies AC14 (complete). |
| T14 | Manual verification gate | T9, T13 | low | (observation only) | Activate venv. Run `--check` → all PASS, exit 0. Run each of the five planners once on `arena_v1` (both `--traffic` and `--no-traffic` for a couple), eyeball `--render` for one replan run and one D\* Lite run to confirm dodging behavior is sane, confirm result dirs/labels. Record outcomes in the PR description. Satisfies AC3, AC6, AC9, AC12. |

## Notes for Implementer

- **The fold's bearing reconstruction must match irsim exactly.** irsim's `Lidar2D` sets `angle_list = np.linspace(angle_min, angle_max, num=number)` with `angle_min = -WrapTo2Pi(angle_range)/2` — spacing is `range/(number-1)`, NOT the `angle_increment = range/number` that `get_scan()` also reports. Reconstruct with `np.linspace`, not `i*angle_increment`, or hits land a fraction of a degree off (compounding at 5 m range). TC28 guards this.
- **Lidar offset is `[0,0,0]`** in `arena_v1` (sensor sits at the robot origin), so the fold maps beams from the robot pose directly. If a future world adds a sensor offset, the fold must compose it — out of scope now, but don't hardcode "no offset" silently; read it if present.
- **The Arena post-processes lidar before the controller sees it:** no-return beams are already `NaN` (not `range_max`). The fold's "non-NaN return" rule is therefore the correct hit test — do not re-threshold against `range_max`.
- **Two waypoint extractors, deliberately.** `a_star_once` keeps the analytic-LOS `plan_waypoints` (so its bytes don't move); replanners + D\* Lite use the grid-native `grid_path_to_waypoints`. Do not route `a_star_once` through the grid extractor — it would change its output and force a re-baseline for no reason.
- **Replan searches from the robot's CURRENT cell**, not `world.start`. `grid_path_to_waypoints` must anchor the first/last waypoints at the actual `start_xy`/`goal_xy`, or the follower will be handed a path that jumps back to the start corner.
- **`astar_search` consumes an `OccupancyGrid`, not a raw array.** The fold returns a bare `np.ndarray`; before calling `astar_search` you MUST re-wrap it: `OccupancyGrid(cells=folded_cells, resolution=grid.resolution, offset=grid.offset)` (reuse the static grid's `resolution`/`offset`). This is load-bearing plumbing the search silently depends on (`astar_search` reads `grid.cells`).
- **Grid-extractor clearance is a different test than analytic LOS.** `segment_is_clear_grid` samples grid cells; `segment_is_clear` (the `_once` path) samples analytic `ObstacleSpec`s. A cost-optimal grid path can still produce a waypoint segment that clips an inflated corner if the grid sampler is coarse — sample at ≤ half-resolution, and let AC5/AC9's clearance assertion catch a too-coarse sampler.
- **Keep-last-path is the whole robustness story for replanners.** Wrap only the `compute_path` call in `act()` with `except (ValueError, RuntimeError)`; do not swallow exceptions from `compute_action_from_state` or the follower — those are real bugs.
- **D\* Lite connectivity MUST match `astar_search`** (same 8 neighbors, same octile step cost, same no-corner-cutting rule) or AC10's "incremental == from-scratch" comparison will fail for the wrong reason. Build the from-scratch oracle in TC36 from the same `astar_search` so the two share the cost model; compare path *cost*, not exact cells (equal-cost ties differ).
- **D\* Lite `k_m` bookkeeping:** bump `k_m += heuristic(s_last, s_start)` on every `move_start`, and `update_cells` before `compute_shortest_path` each tick. Getting the key-modifier order wrong yields a subtly suboptimal path that still "looks" fine end-to-end — TC36 is the real guard, so make it adversarial (block a cell that's on the current optimal path).
- **Determinism is relaxed but not sabotaged.** Don't introduce `set`/`dict` iteration over cells in a way that changes results run-to-run; prefer sorted/stable iteration in the fold and the changed-cell diff so traces stay reproducible in practice even though no TC gates it.
- **`build_controller` is the single validation choke point** for `--replan-k`. Keep argparse `choices=list(ALGORITHMS)`; do the family/K cross-validation in `build_controller` (raising `ValueError`) so both `run_episode` and `run_experiment` get identical behavior from one place. The runner maps that `ValueError` to exit 2 (config error), distinct from a planner `ValueError` at `reset()` (which is `planner_error`, exit 0) — keep these two error channels straight.
- **`run_experiment` label parity:** the batch runner must compute `algorithm_label` the same way `run_episode` does (forward `--replan-k`, build the same dir) so the manifest lands beside the episodes. Re-use the imported helper; don't re-derive the label string.
- **Performance:** Dijkstra (`h=0`) and per-step replans on the 500×500 grid (50 m / 0.1 m) are heavier than A\*-once. Mission grants "no compute budget" and `wallclock_per_step` is a freebie; low-K replan runs will be slow in wall-clock — that's expected, run them with `run_experiment --jobs N` if needed. Do not "optimize" by shrinking the grid resolution (changes the comparison substrate).
- **PR hygiene** (user-global rules): commits/PR read as human-authored — no AI attribution, no "Generated with Claude Code", no em-dashes where a comma works. Match the repo's `type(scope): subject` commit voice (`git log`). Two PRs are fine (Group A, then Group B) or one PR with two commits; Group A must be green before Group B lands.
- **Rollback:** delete `planners/_grid.py`, `planners/dijkstra.py`, `planners/d_star_lite.py`; revert `planners/_types.py`/`a_star.py`/`__init__.py`, the `run_episode.py`/`run_experiment.py` redesign, the `manual_astar.py` additions, the TC28–TC37 additions, and the doc edits. Repo returns to its Phase 3 state (`a_star_once` only).
- **What this plan deliberately does NOT do:** reactive controllers (DWA/APF), sampling planners (RRT/RRT\*) and their RNG plumbing, the 6b K-sweep loop, Phase 4 aggregation, or Phase 5 plotting. If the executor finds themselves writing a velocity-sampling controller, an RRT tree, or a K-sweep driver, stop — that's a different plan.
