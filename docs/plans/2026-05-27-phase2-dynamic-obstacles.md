# Phase 2 — Dynamic Obstacles (Crossing Traffic) Plan

**Goal:** Land the crossing-traffic substrate so every planner in Mission.md runs against a moving, seeded ~20-obstacle population. Static layout and seed plumbing already exist; Phase 2 plugs the spawner behind the `initial_dynamic_snapshot` seam, makes irsim's lidar see the obstacles natively, and proves end-to-end determinism with traffic enabled.

**Approach:** Add an `arena/dynamic.py` module that defines a `DynamicObstacleState` frozen dataclass and a `TrafficSpawner` that owns the live obstacle population. `Arena.__init__` gains a `traffic: bool = False` flag; when True, the spawner instantiates 20 obstacles during `reset()` via `env.create_obstacle(kinematics={'name': 'omni'}, ...) + env.add_object(...)` so irsim's lidar and collision detection pick them up natively. Each `arena.step()` advances obstacles, despawns those that exit the arena, refills to target via `env.delete_object()` + new spawns, then calls `env.step([action])`. The `initial_dynamic_snapshot` property narrows to `tuple[DynamicObstacleState, ...]`. The runner flips its default to traffic ON, the trace JSONL gains a conditional `dynamic_obstacles_sha256` field, and five new test cases (TC17–TC21) extend `arena/arena.py --check` from 17 PASS to 22 PASS.

## Scope

- **In scope:**
  - `arena/dynamic.py` — `DynamicObstacleState` frozen dataclass + `TrafficSpawner` class (init population, per-tick motion + despawn + refill, deterministic sampling from `traffic_rng`/`motion_rng`).
  - `arena/arena.py` — `traffic: bool = False` kwarg on `Arena.__init__`; spawner instantiation in `__init__`; per-tick advance + refill in `step()` BEFORE `env.step()`; reset() re-initializes the spawner; `initial_dynamic_snapshot` returns `tuple[DynamicObstacleState, ...]`; `info.dynamic_obstacle_count` becomes non-zero when traffic is on; `info.dynamic_obstacles_sha256: str | None` exposes the per-tick state hash; TC17–TC21 added to `_run_checks`.
  - `runners/run_episode.py` — `--traffic / --no-traffic` flag (default `--traffic`); pass `traffic=` to `Arena`; trace JSONL adds `dynamic_obstacles_sha256` only when traffic is enabled; TC14 in `arena/arena.py` updated to invoke the runner with `--no-traffic` so its existing A*-success metric bounds still hold; TC15 same change. **World-stem partitioning** (NEW): output path changes from `<results-dir>/<algorithm>/<seed>.{json,trace.jsonl}` to `<results-dir>/<world_stem>/<algorithm>/<seed>.{json,trace.jsonl}` where `<world_stem> = Path(args.world).stem`. Prevents same-seed runs on different worlds (e.g. `arena_v1.yaml` vs `arena_v2_hard.yaml`) from silently overwriting each other.
  - `planners/_types.py` — narrow `initial_dynamic_snapshot` parameter type in the `PathPlanner` Protocol from `tuple` to `tuple[DynamicObstacleState, ...]` (re-imported from `arena.dynamic`).
  - `CLAUDE.md` — new "Phase 2 — dynamic obstacles" section: `traffic=True` semantics, the `arena/dynamic.py` API surface, the runner default change, the new `<results-dir>/<world_stem>/<algorithm>/<seed>` results layout (and migration note for any human-stored results from Phase 1), and one-line summaries of TC17–TC23. The existing "## The episode runner (Phase 1)" section's path examples and "Results layout" bullets need updating to the new layout.
- **Out of scope:**
  - Dynamic-aware planners (`a_star_replan_K`, DWA, APF, D* Lite, RRT*) — Phase 6. `AStarOncePlanner.plan()` continues to ignore `initial_dynamic_snapshot` and `lidar0`.
  - Reactive-planner Protocol (`plan() -> (v, ω)`) — Phase 6.
  - K-sweep replan plumbing — Phase 6b.
  - `results/plot.py` — Phase 5.
  - 50-seed master list / `SeedSequence(MASTER).spawn(50)` runner helper — Phase 3 (the same PR can absorb it if cheap, but Phase 2 does NOT block on it).
  - Spawn-point distribution research (edge-biased, center-biased) — Phase 2 ships the simple "uniform along perimeter" rule and revisits only if traffic feels degenerate (Mission.md §Phase 2 explicit guidance).
  - Per-obstacle motion noise — straight-line motion is deterministic given init state; `motion_rng` is plumbed but currently unused.
  - Refactoring `manual_astar.py` or splitting `WorldModel` into a shared package — Phase 6.
  - pytest infrastructure — TC-pattern in `arena/arena.py --check` continues.

## Decisions

- **Obstacle injection via `env.create_obstacle(kinematics={'name': 'omni'}, ...) + env.add_object(obs)`** — Native irsim API; lidar and collision detection work without bespoke code. `ObstacleOmni` is deprecated per `.venv/Lib/site-packages/irsim/world/obstacles/obstacle_omni.py:9-12`; the factory route is the supported path. Rejected: (a) pre-declare 20 placeholder circles in YAML and teleport (couples spawner count to YAML, brittle parking state), (b) manual rasterization (bypasses irsim's collision flag — we'd lose the very signal Mission.md tells us to use).
- **Spawn distribution: uniform along total perimeter (4 × 50 = 200 m)** — Sample `t ∈ [0, 200)` via `traffic_rng.uniform`, map to (edge, position-along-edge) deterministically. Rejected: per-edge uniform (slight corner bias), diagonal-biased (premature optimization; Mission.md says start simple).
- **Heading distribution: uniform over the 180° inward-facing half-cone** — South edge → heading in (0, π); east edge → (π/2, 3π/2); north → (π, 2π); west → (-π/2, π/2). Guarantees the obstacle enters the arena rather than immediately exiting. Rejected: full 360° (~50% wasted spawns) and center-biased (artificial).
- **Refill cadence: immediate, same-tick** — Each `arena.step()` first advances + despawns, then loops `while len(alive) < TARGET_POPULATION: spawn()`. Literal reading of Mission.md "respawn as soon as old ones exit." The robot cannot game the experiment by waiting. Rejected: Poisson (variance dips population below 20) and fixed-interval (cannot guarantee target).
- **Initial population fully alive at t=0; all 20 spawned on edges with inward headings** — `reset()` runs the same spawn rule as `step()` until population == 20. The lidar sees inbound traffic on the very first scan. `initial_dynamic_snapshot` returns the t=0 set. Rejected: trickle-in (empty snapshot, violates Mission.md's "planner knows initial positions") and interior-spawn at t=0 (needs separate sampler; not worth the complexity for the t=0 boundary case alone).
- **Spawn-overlap policy: reject + resample, max 20 attempts** — Reject any sampled spawn whose obstacle circle (r=0.3 m) overlaps a static obstacle (using `manual_astar.point_to_obstacle_distance`) or the robot start within a 1.0 m safety buffer. Dynamic-vs-dynamic overlap at spawn is ALLOWED (Mission.md: "obstacles pass through each other"). After 20 failed attempts for a single spawn, give up for this tick (population temporarily < 20; refilled next tick). Bounded RNG draws keep determinism predictable. Rejected: never check (bad-luck spawn on robot start) and unbounded retry (infinite-loop risk in degenerate arenas).
- **Snapshot type: `tuple[DynamicObstacleState, ...]`** with `DynamicObstacleState` a frozen dataclass of fields `(id: int, x: float, y: float, vx: float, vy: float, radius: float)` — Matches the project's dataclass-heavy style; named fields beat magic-column numpy arrays; immutable. Phase 6 dynamic-aware planners read by attribute. Rejected: `(N, 5)` numpy array (column-order is foot-gun) and dict (mutable, no type-checker).
- **Per-tick determinism telemetry: `dynamic_obstacles_sha256` on `EpisodeInfo` and trace JSONL line** — Mirror the `lidar_sha256` pattern. Hash bytes of a stable-order `(N, 5)` float64 array `[x, y, vx, vy, radius]` sorted by obstacle `id`. ~64 B/line. Adding the per-step state to the trace is the cheap way to catch a determinism drift at the cause site (which obstacle, which tick) rather than just the symptom (lidar hash diverged). Rejected: full state dump (~5 KB/line) and lidar-hash-only (loses cause/symptom split).
- **Trace JSONL key is conditional on `traffic`** — `dynamic_obstacles_sha256` is appended ONLY when `Arena` was constructed with `traffic=True`. Phase 1's TC14 schema check (`set(row) == TC14_TRACE_REQUIRED_KEYS`) is updated to use a derived set: `TC14_TRACE_REQUIRED_KEYS | {'dynamic_obstacles_sha256'}` when the subprocess was invoked with `--traffic`. TC14 itself stays on `--no-traffic` so its A*-success bounds still hold.
- **Arena flag, NOT a new YAML** — `Arena(yaml, seed, traffic=True/False, ...)`. `arena_v1.yaml` is unchanged; same canonical world. Avoids YAML schema growth + the "which YAML for which test" cognitive load. Rejected: separate `arena_v2.yaml` and always-on (rewrites TC4/TC13 because traffic could interfere with deliberate-crash drives).
- **Runner defaults to `--traffic`** — `python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml` now runs WITH traffic by default. Phase 6 experiments (Mission.md headline scatter plot) require traffic; making it the default reduces footgun. TC14 and TC15 are updated to explicitly pass `--no-traffic` so their existing A*-success expectations still hold (A* cannot dodge dynamic obstacles, so under traffic it will crash on most seeds — meaningful for the Phase 5 plot, fatal for Phase 1's smoke test). Rejected: traffic=OFF default (mismatches the experiment's actual workload) and required-flag (extra typing for the common case).
- **Results path: `<results-dir>/<world_stem>/<algorithm>/<seed>.{json,trace.jsonl}`** — World stem is the TOP-LEVEL subdirectory under `results/`, algorithm nests beneath. Rationale: with `arena_v1.yaml`, `arena_v2_hard.yaml`, and `arena_no_path.yaml` now in the repo (and more worlds expected as the experiment grows), the Phase 1 layout `<results-dir>/<algorithm>/<seed>.{json,trace.jsonl}` silently overwrites across worlds — running seed=42 on `arena_v1` then on `arena_v2_hard` clobbers the first run with the second. World-on-top ordering makes "compare all algorithms within one world" a single `ls results/<world>/` and matches how Phase 5's plot.py will partition data: one (algorithm × seed) scatter per world. Rejected alternatives: `<results-dir>/<algorithm>/<world_stem>/<seed>.…` (less natural for the cross-algorithm comparison the experiment is built to produce); flat `<results-dir>/<algorithm>/<world_stem>_<seed>.json` (brittle string parsing in plot.py, harder to glob per world); keep current layout + require unique seeds per world (footgun, no enforcement).
- **Reuse irsim `collision_flag` for dynamic-obstacle hits, no extra Python check** — Once obstacles are real `obstacle_list` entries, irsim's collision detection fires on robot-vs-dynamic contact identically to robot-vs-static. TC19 verifies this end-to-end (teleport robot into a dynamic obstacle's predicted path). If TC19 reveals tunneling (fast obstacle skips over robot in one tick), file a follow-up; do not add a defensive Python overlap check pre-emptively.
- **Despawn boundary: obstacle center outside `[-DESPAWN_BUFFER, 50 + DESPAWN_BUFFER]` on either axis, with `DESPAWN_BUFFER = 0.5 m`** — Half-radius beyond the wall so an obstacle that just crossed the boundary cannot still emit lidar returns inside the arena. The buffer also avoids border-flicker spawn/despawn loops.
- **Spawner runs in `arena.step()` BEFORE `env.step([action])`** — Order: (1) advance each obstacle's position by `vx*dt, vy*dt`, (2) despawn out-of-bounds via `env.delete_object(target_id)`, (3) refill via `env.create_obstacle + env.add_object` until population == 20, (4) call `env.step([action])` so the lidar samples post-move positions on the same tick. Rejected: spawner after `env.step()` (lidar lags reality by one tick); irsim's native per-obstacle velocity integration via `env.step([action_robot, *velocities])` (would let irsim integrate kinematics — but requires injecting a per-tick velocity action vector keyed by id, which couples spawner state to irsim's action-step ordering and complicates the determinism contract; manual `x += vx*dt` keeps a single code path responsible for obstacle motion).
- **`motion_rng` plumbed but unused in Phase 2** — Straight-line motion is deterministic given init state; no per-tick noise to draw. Kept in `__init__` and re-derived in `reset()` so a Phase 2b extension (e.g. heading jitter, speed perturbation) can attach without an interface change. `traffic_rng` is consumed at spawn time only. **Invariant for Phase 2**: `motion_rng` is NEVER drawn from in Phase 2 spawner code, so AC4/AC12 hash sequences stay stable across any later Phase 2b refactor that only adds new draws via `motion_rng`.
- **Reset must delete previously-spawned dynamic obstacles BEFORE re-spawning** — `irsim.env.reset()` resets every object in `env.objects` to its `_init_state` (verified at `.venv/Lib/site-packages/irsim/env/env_base.py:_reset_all`); it does NOT delete dynamically-added entries. If the spawner naively calls `initialize()` again after `env.reset()`, the population doubles (existing 20 obstacles + 20 freshly spawned). `Arena.reset()` MUST iterate `self._spawner.live_ids` and call `env.delete_object(id)` for each BEFORE invoking `spawner.initialize()`. Phase 0's TC7 (reset-after-done) becomes a load-bearing guard against this regression. See Notes for Implementer.
- **`initial_dynamic_snapshot` pre-`reset()` returns `()` even when `traffic=True`** — Snapshot is only populated by `spawner.initialize()` inside `reset()`. Between `Arena(..., traffic=True)` construction and the first `reset()`, the property returns the empty tuple to match Phase 0 contract. Documented in AC.
- **Phase 2 ships as one PR** — `dynamic.py` + Arena integration + runner integration + TC17-TC21 + docs land together. Tight coupling (the runner test depends on the spawner; the TCs depend on the runner; the docs depend on the final API) makes a multi-PR split add only noise.
- **No `*_replan_K` planner included** — Mission.md tags those as Phase 6. Phase 2 only proves the substrate. The headline insight of the experiment (Phase 7) compares replan strategies; that comparison cannot run until the substrate works.

## Acceptance Criteria

- [ ] **AC1:** `Arena('arena/arena_v1.yaml', seed=42, traffic=True)` constructs without error and `arena.reset()` returns an `EpisodeInfo` with `dynamic_obstacle_count == 20` and `lidar_status == "ok"`. The lidar-difference check is **NOT** asserted on tick 0 (lidar range is 5 m per `arena_v1.yaml`; uniform-on-perimeter spawn produces obstacles within 5 m of the robot start only ~40% of the time per seed). Instead, after 600 zero-action ticks at `traffic=True` vs the same 600 ticks at `traffic=False`, the union of per-beam differences across that window is non-empty (i.e., over a 60-second window at least one tick has a beam that differs). This guarantees the dynamic obstacles ARE producing lidar effects without flaking on tick-0 luck.
- [ ] **AC2:** `arena.initial_dynamic_snapshot` returns a `tuple` of length 20 whose entries are `DynamicObstacleState` instances; each instance has finite `x, y, vx, vy` and `radius == 0.3`; per-obstacle speed `sqrt(vx² + vy²)` is in `[0.3, 1.5]`; all `(x, y)` lie on or near (within `1e-6` m) one of the four arena edges; the inward-half-cone heading rule holds for each spawn (verifiable per-edge: south spawns have `vy > 0`, etc.).
- [ ] **AC3:** Across 600 consecutive `arena.step(np.array([[0.0], [0.0]], dtype=float))` calls with `traffic=True`, `info.dynamic_obstacle_count` is `20` on every tick (i.e., the spawner refills synchronously). Population dipping below 20 on any tick fails the AC.
- [ ] **AC4:** Two same-seed `Arena(yaml, seed=42, traffic=True)` instances, each driven through 200 zero-action steps, produce byte-identical sequences of `info.dynamic_obstacles_sha256` strings. The byte-identity of `lidar_sha256` sequences is a STRETCH GOAL — verified in T1's sanity step before this AC is declared satisfied. If irsim's STRtree query order proves non-deterministic across spawn/despawn cycles (the spec author's open concern), AC4 falls back to: "dynamic_obstacles_sha256 sequences byte-identical; lidar sequences equal up to per-tick set of finite beam returns (sorted)." T1 ships whichever guarantee actually holds; T6's verification step records which.
- [ ] **AC5:** `Arena.close()` cleanly tears down a traffic-enabled environment; a subsequent `Arena(..., traffic=True)` in the same Python process can be constructed and reset without irsim "object name already exists" errors. (Verifies the spawner gives unique object names and the cleanup path runs.)
- [ ] **AC6:** `python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml` (no `--no-traffic`) runs end-to-end, exits 0, and writes `results/arena_v1/a_star_once/42.json` and `results/arena_v1/a_star_once/42.trace.jsonl` (note the new world-stem subdirectory). Each trace line has the seven Phase 1 keys PLUS `dynamic_obstacles_sha256` (8 keys total).
- [ ] **AC7:** `python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml --no-traffic` produces 7-keys-per-line trace JSONL with no `dynamic_obstacles_sha256` field at `results/arena_v1/a_star_once/42.trace.jsonl`. Two such `--no-traffic` runs in separate tempdirs are byte-identical (TC15 mechanism). The "byte-identical to a pre-Phase-2 Phase 1 baseline trace" comparison is NOT asserted — no pre-Phase-2 baseline is pinned in the repo, and the simulation is unchanged for `--no-traffic` so byte-identity holds by construction; the test is "two same-seed `--no-traffic` runs produce identical bytes" not "matches a stored baseline."
- [ ] **AC15:** World-stem partitioning — Running `... --seed 42 --world arena/arena_v1.yaml` followed by `... --seed 42 --world arena/arena_v2_hard.yaml` in the same `--results-dir` produces FOUR distinct files (`results/arena_v1/a_star_once/42.{json,trace.jsonl}` and `results/arena_v2_hard/a_star_once/42.{json,trace.jsonl}`); neither run overwrites the other. Running `... --world arena/arena_no_path.yaml --seed 0` lands the metrics JSON at `results/arena_no_path/a_star_once/0.json` (no trace JSONL, per the existing planner-failure contract).
- [ ] **AC8:** `python arena/arena.py arena/arena_v1.yaml --check` reports **24 PASS** lines (17 existing + TC17 + TC18 + TC19 + TC20 + TC21 + TC22 world-stem partitioning + TC23 import-cycle guard) and exits 0 in under 120 s wallclock on the dev laptop.
- [ ] **AC9:** TC17 (init population) — `Arena('arena/arena_v1.yaml', seed=0, traffic=True).reset()` produces `info.dynamic_obstacle_count == 20` and `len(arena.initial_dynamic_snapshot) == 20`; every obstacle is on an edge with inward heading.
- [ ] **AC10:** TC18 (refill) — Stepping a `traffic=True` Arena for `ceil(50 / 0.3 / dt) + 50` zero-action ticks (enough time for the slowest obstacle to traverse the arena and despawn at least once) confirms `info.dynamic_obstacle_count == 20` at every tick AND at least one despawn has occurred (verifiable by tracking obstacle ids and seeing at least one id disappear from the live set).
- [ ] **AC11:** TC19 (dynamic crash) — Construct `traffic=True` Arena (`seed=2`); ALWAYS use the deterministic injection path: call `spawner._inject_for_test(x=3.0, y=2.0, vx=-1.0, vy=0.0, radius=0.3)` to place an obstacle 1.0 m east of the robot start `(2, 2)` moving west at 1.0 m/s. Step zero actions; with robot radius 0.2 m and obstacle radius 0.3 m, the contact-distance threshold is 0.5 m, so collision occurs when the obstacle has moved `(1.0 - 0.5) = 0.5 m`, i.e., 5 ticks at dt=0.1. Assert `info.crashed is True` within 20 ticks (4× safety margin). Separately, assert that the natural `seed=2` snapshot contains at least one obstacle whose straight-line trajectory passes within 2.0 m of `(2, 2)` within the next 100 ticks (so the test also documents that the natural-traffic crash path is reachable — without depending on it for the assertion).
- [ ] **AC12:** TC20 (traffic determinism, `--check` TC form) — TC20 IS the executable form of AC4. Two same-seed Arenas, each driven 200 ticks with zero actions, produce identical per-tick `info.dynamic_obstacles_sha256` sequences. Lidar byte-identity is verified per AC4's fallback rule.
- [ ] **AC13:** TC21 (snapshot shape) — `Arena('arena/arena_v1.yaml', seed=5, traffic=False).initial_dynamic_snapshot == ()` both pre-reset AND post-reset. `Arena(..., seed=5, traffic=True).initial_dynamic_snapshot == ()` pre-reset (snapshot only built in `reset()`). After `arena.reset()`, the same property returns `tuple[DynamicObstacleState, ...]` of length 20; each instance is frozen (`dataclasses.is_dataclass`); attempting to mutate a field raises `dataclasses.FrozenInstanceError`. `snapshot[0].radius == 0.3`.
- [ ] **AC14:** `CLAUDE.md` has a new "Phase 2 — dynamic obstacles" section (~15-25 lines) covering: the `Arena(..., traffic=True)` flag, the `arena/dynamic.py` API surface (DynamicObstacleState fields and TrafficSpawner public methods), the runner default change to `--traffic`, the `dynamic_obstacles_sha256` trace key, and one-line summaries of TC17–TC21.

## Data Model

```python
# arena/dynamic.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

TARGET_POPULATION = 20
OBSTACLE_RADIUS = 0.3
SPEED_MIN_FACTOR = 0.3
SPEED_MAX_FACTOR = 1.5
ROBOT_TOP_SPEED = 1.0           # mirrors manual_astar.MAX_LINEAR_SPEED
SPAWN_OVERLAP_BUFFER = 1.0      # m, robot-start clearance
DESPAWN_BUFFER = 0.5            # m, beyond arena boundary
SPAWN_MAX_ATTEMPTS = 20         # per spawn, then give up for this tick
DYNAMIC_OBSTACLE_NAME_FMT = "traffic_{idx}"  # globally unique per Arena lifetime


@dataclass(frozen=True)
class DynamicObstacleState:
    id: int          # irsim object id; unique within an Arena
    x: float
    y: float
    vx: float        # m/s, world frame
    vy: float        # m/s, world frame
    radius: float    # always OBSTACLE_RADIUS in Phase 2; kept for forward compat


class TrafficSpawner:
    """Maintains a TARGET_POPULATION population of straight-line crossing obstacles.

    Lifecycle:
      __init__(env, robot, traffic_rng, motion_rng, dt, arena_w, arena_h, static_obstacles)
      initialize() -> tuple[DynamicObstacleState, ...]    # called from Arena.reset()
      step() -> tuple[DynamicObstacleState, ...]          # advance + despawn + refill
      snapshot() -> tuple[DynamicObstacleState, ...]      # current state, no mutation
      state_sha256() -> str                               # deterministic hex of snapshot
      close() -> None                                     # idempotent cleanup
    """
    ...
```

```python
# arena/arena.py — only the diff from Phase 1
class Arena:
    def __init__(
        self,
        yaml_path: str | Path,
        seed: int,
        render: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        traffic: bool = False,                       # NEW
    ) -> None: ...

    @property
    def initial_dynamic_snapshot(self) -> tuple[DynamicObstacleState, ...]: ...

@dataclass(frozen=True)
class EpisodeInfo:
    sim_time: float
    step_idx: int
    crashed: bool
    timed_out: bool
    reached_goal: bool
    distance_to_goal: float
    wallclock_per_step: float
    dynamic_obstacle_count: int                       # was always 0; now 0 or 20
    lidar_status: str
    dynamic_obstacles_sha256: str | None              # NEW; None when traffic=False
```

## API Contracts

`TrafficSpawner.step()` semantics:

```
Input:  none (consumes traffic_rng + dt internally)

Side effects, in order:
    1. Advance each live obstacle: x += vx*dt, y += vy*dt; write back via irsim object handle.
    2. Despawn: for any obstacle with center outside [-DESPAWN_BUFFER, arena_w + DESPAWN_BUFFER]
       (and same for y), call env.delete_object(obstacle.id); remove from live set.
    3. Refill: while len(live) < TARGET_POPULATION:
         (edge, t_along_edge) := sample uniform-on-perimeter via traffic_rng
         heading              := sample uniform from inward half-cone for that edge
         speed                := sample uniform in [SPEED_MIN_FACTOR, SPEED_MAX_FACTOR] * ROBOT_TOP_SPEED
         if overlaps static obstacle or robot start (buffer):
             retry up to SPAWN_MAX_ATTEMPTS; if all fail, break (refill rest next tick)
         else:
             obs := env.create_obstacle(kinematics={'name':'omni'},
                                        shape={'name':'circle','radius':OBSTACLE_RADIUS},
                                        state=[x, y, 0],
                                        name=DYNAMIC_OBSTACLE_NAME_FMT.format(idx=self._next_idx))
             env.add_object(obs)
             cache (id, vx, vy) for snapshot construction
             self._next_idx += 1

Output: snapshot tuple of DynamicObstacleState

Determinism:
    traffic_rng is consumed in a fixed order per tick: (advance reads no RNG; despawn reads no RNG;
    refill draws perimeter-position, then heading, then speed, then overlap-retries — in that order,
    per spawn). Documented in a one-line comment at the call site.
```

`Arena.step(action)` semantics — diff from Phase 1:

```
Pre-step (new, when traffic=True):
    self._spawner.step()           # advance + despawn + refill, returns new snapshot
    self._last_snapshot = snapshot
    self._last_sha256   = self._spawner.state_sha256()

Existing irsim call:
    self._env.step([action])

Info construction (new fields):
    info.dynamic_obstacle_count = len(self._last_snapshot) if traffic else 0
    info.dynamic_obstacles_sha256 = self._last_sha256 if traffic else None

Error cases (additive):
    if traffic and env.add_object fails (name collision): bubble as ArenaRuntimeError.
```

Runner CLI diff:

```
--traffic / --no-traffic     mutually exclusive; default --traffic
                             passed to Arena(... , traffic=<bool>)

Trace JSONL line (per step):
    when traffic=False: 7 keys, byte-identical to Phase 1
    when traffic=True : 7 keys + "dynamic_obstacles_sha256": str
    sort_keys=True is preserved, so the 8th key sorts in deterministically.

Metrics JSON: unchanged 7 keys (planner_error stays; no new field).
```

## Error Handling

- **`traffic=True` but YAML is malformed (e.g. lidar count wrong)** — `Arena.__init__` raises `ArenaConfigError` BEFORE constructing the spawner. The spawner is not partially initialized.
- **Spawn overlap with static obstacles exhausts SPAWN_MAX_ATTEMPTS** — Spawner gives up for this tick and emits no warning (the next tick will retry). Documented in `Notes for Implementer`. If population stays < 20 for many consecutive ticks, the underlying issue is arena saturation — verify via TC18.
- **`env.add_object` raises `ValueError("Object name '...' already exists")`** — Spawner uses a monotonically increasing index in `DYNAMIC_OBSTACLE_NAME_FMT`, so this should not happen in practice. If it does, surface as `ArenaRuntimeError` — programmer bug.
- **`env.delete_object(id)` raises (id not found)** — Spawner only deletes ids it tracks in its live set. If deletion fails, raise `ArenaRuntimeError` (irsim state desync).
- **Robot collides with a dynamic obstacle** — irsim sets `robot.collision_flag = True`; existing pre/post-step guard in `arena.step()` already handles flag latching. No new code; covered by TC19.
- **`render=True` with `traffic=True`** — Each dynamic obstacle's `_init_plot` runs at `env.add_object` time (per `env_base.py:917-918`); no new render code needed. Performance: 20 circles repainted per tick should be negligible. Not asserted, but verifiable in T-Verify.
- **`Arena.close()` with traffic still alive** — `env.end()` tears down the whole env including dynamic obstacles. Spawner's `close()` is idempotent and clears its internal list; called from `Arena.close()` even when `traffic=False` (it's a no-op).

## Testing Strategy

**Levels:** Unit (spawner sampling math, snapshot shape, deterministic state_sha256), Integration (Arena with traffic=True over many ticks, runner end-to-end with --traffic), Regression (Phase 1 TCs still PASS with --no-traffic).

All tests live as `TCi` functions inside `arena/arena.py --check` per project convention. No pytest.

| ID   | Test Case | Type | Expected Behavior |
|------|-----------|------|-------------------|
| TC17 | `Arena('arena/arena_v1.yaml', seed=0, traffic=True).reset()` | Integration | `info.dynamic_obstacle_count == 20`; `len(arena.initial_dynamic_snapshot) == 20`; every entry is a `DynamicObstacleState`; every entry's position is on a perimeter edge (within 1e-6 m); each entry's heading is in the inward half-cone for its edge; each entry's speed in `[0.3, 1.5]`. |
| TC18 | Refill maintained over a full-traversal window | Integration | Construct `traffic=True` Arena seed=1; run `ceil(50/0.3/dt) + 50` zero-action steps; assert `info.dynamic_obstacle_count == 20` on every tick AND the set of live obstacle ids changes between tick 0 and the final tick (i.e., at least one despawn-and-respawn happened). |
| TC19 | Robot-vs-dynamic collision flag | Integration | Construct `traffic=True` Arena seed=2; pick the dynamic obstacle whose forward trajectory comes closest to the robot start within the next 100 ticks; teleport the robot to a pose that intersects that obstacle's path in `<=50` ticks; step zero actions until `info.crashed` or 100 ticks. Assert `info.crashed is True` within budget. If no dynamic obstacle's trajectory comes within `<2 m` of the robot start (rare), the test falls back to inserting a synthetic obstacle directly via `spawner._inject_for_test(x=2.5, y=2.0, vx=-1.0, vy=0.0)` and asserts the same outcome. |
| TC20 | Traffic determinism — per-tick hashes | Integration | Two `Arena(..., seed=3, traffic=True)` instances, each driven 200 zero-action steps; record `info.dynamic_obstacles_sha256` per tick and `hashlib.sha256(lidar.tobytes()).hexdigest()` per tick; assert both per-tick lists are equal between the two instances. |
| TC21 | Snapshot shape, type, immutability | Unit | `Arena(..., seed=4, traffic=False).initial_dynamic_snapshot == ()`. `Arena(..., seed=4, traffic=True).initial_dynamic_snapshot` is `tuple[DynamicObstacleState, ...]` of length 20. `dataclasses.is_dataclass(snapshot[0])`; the dataclass is frozen (assigning to a field raises `dataclasses.FrozenInstanceError`). `snapshot[0].radius == 0.3`. |
| TC22 | World-stem partitioning end-to-end | Integration | Subprocess-invoke the runner twice with `--seed 42`, the same `--results-dir <tmpdir>`, and two different `--world` values (e.g. `arena/arena_v1.yaml` and `arena/arena_v2_hard.yaml`, both with `--no-traffic` so A* succeeds on both). Assert: `<tmpdir>/arena_v1/a_star_once/42.json` exists, `<tmpdir>/arena_v2_hard/a_star_once/42.json` exists, both trace JSONLs exist at their respective paths, AND `json_a != json_b` (different worlds at the same seed must produce different `path_length` / `time_to_goal`). |
| TC23 | Import-cycle guard | Unit | Subprocess `python -c "import planners; import arena.arena"` exits 0; subprocess `python -c "import arena.arena; import planners"` exits 0. Catches accidental cycles introduced by `arena.dynamic → manual_astar` (lazy) and `planners._types → arena.dynamic` (direct) edges. |

**Test data:** All tests use `arena/arena_v1.yaml`. TC19 may use `spawner._inject_for_test` (a test-only helper marked with an underscore prefix; the spec authorizes adding it to `arena/dynamic.py`).

**Run commands:**

```powershell
.venv\Scripts\Activate.ps1
python arena/arena.py arena/arena_v1.yaml --check          # 24 PASS (17 existing + TC17-TC23)
python arena/arena.py arena/arena_v1.yaml --render         # visible smoke (Phase 0 behavior unchanged)
python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml
                                                            # NEW DEFAULT: traffic ON; A* will crash on most seeds
                                                            # output:  results/arena_v1/a_star_once/42.{json,trace.jsonl}
python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v2_hard.yaml
                                                            # different world, same seed -> different result file
                                                            # output:  results/arena_v2_hard/a_star_once/42.{json,trace.jsonl}
python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml --no-traffic
                                                            # same-seed --no-traffic runs are byte-identical (TC15)
```

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T0 | **Create feature branch** `phase2-dynamic-obstacles` from main (project CLAUDE.md rule: "Always create feature branches for new work. Never commit directly to main unless explicitly told to."). All Phase 2 commits land on this branch. | — | low | (git) | One command: `git checkout -b phase2-dynamic-obstacles`. T1 starts on this branch. |
| T1 | **Implement `arena/dynamic.py`**: `DynamicObstacleState` frozen dataclass with fields `(id, x, y, vx, vy, radius)`. Module-level constants per the Data Model section. `TrafficSpawner` class with: `__init__(env, robot, traffic_rng, motion_rng, dt, arena_w, arena_h, static_obstacles)`; `live_ids` property (frozenset of currently-spawned obstacle ids; used by `Arena.reset()` to delete stale entries before re-init); `initialize() -> tuple[DynamicObstacleState, ...]` (spawns to TARGET_POPULATION; reused by Arena.reset()); `step() -> tuple[DynamicObstacleState, ...]` (advance → despawn → refill in that order); `snapshot()` and `state_sha256()` (deterministic — sort by id, build (N,5) float64 array of `[x,y,vx,vy,radius]`, hash bytes); `close()` (idempotent: delete every tracked id from env, clear caches); `_inject_for_test(x, y, vx, vy, radius)` test helper that bypasses `traffic_rng` (does NOT consume any RNG draw). Use `manual_astar.point_to_obstacle_distance` for the static-overlap check (import lazily to avoid cycles, mirroring the TC10 sys.path pattern). Use `env.create_obstacle + env.add_object` for new obstacles; `env.delete_object` for despawn. Names follow `DYNAMIC_OBSTACLE_NAME_FMT` with a monotonically increasing index. **Sanity check at the end of T1:** run a 200-tick scratch script that calls `initialize() → step() × 200` twice with the same seed, and assert `(a) state_sha256 sequences are byte-identical; (b) lidar bytes are byte-identical; (c) the set of live obstacle ids on tick 200 changes between two different seeds (spawner is not seed-invariant).` Record the result in a comment at the top of `dynamic.py`: lidar-byte-identity confirmed OR falls back to id+position equality (per AC4 fallback). Satisfies AC2, AC3, AC9, AC10 substrate; provides the snapshot+sha256 used by AC4/AC12. | T0 | high | `arena/dynamic.py` (new) | This is the single largest task; do not split. Critical determinism rule: traffic_rng is consumed in a FIXED order per spawn — (perimeter t, heading, speed, then per overlap-retry repeat). Document the order in a one-line comment at the call site so a future maintainer cannot accidentally reorder draws. The `state_sha256()` MUST sort by `id` before hashing — Python dict insertion order or any other implicit order is a determinism foot-gun. **`motion_rng` MUST NOT be drawn from in Phase 2** (any future change that adds a draw must update AC4/AC12 hash baselines simultaneously). **Spawn rejection math:** two 30 m walls block ~60 m of inner perimeter, but the perimeter is the outer 4×50 = 200 m loop — walls do not intersect the perimeter, so wall-rejection is rare. Robot-start rejection clears a ~1.2 m disc near `(2, 2)`, intersecting the perimeter only near the SW corner (a ~3 m arc). Acceptance rate ≥ ~98% per attempt; `SPAWN_MAX_ATTEMPTS = 20` is comfortably over-budgeted. Do NOT lower the constant. **`point_to_obstacle_distance` returns SIGNED distance** for some obstacle kinds (negative inside); the rejection test `< OBSTACLE_RADIUS + SPAWN_OVERLAP_BUFFER` covers both branches correctly. |
| T2 | **Wire `TrafficSpawner` into `Arena`**: add `traffic: bool = False` kwarg on `__init__`; instantiate `self._spawner` after irsim env construction (spawner is constructed but `initialize()` is NOT called yet); gather static-obstacle specs from the loaded YAML for the overlap check (reuse `manual_astar.load_world` lazily); **`reset()` ordering — CRITICAL** (in this order): (1) call `env.reset()` (this resets all objects to `_init_state` and runs irsim's internal warm-up step; dynamically-added obstacles are NOT removed by this), (2) re-derive `traffic_rng + motion_rng` from `master_seed` via `SeedSequence.spawn(2)`, (3) **DELETE every id in `self._spawner.live_ids` via `env.delete_object(id)` to clear stale dynamics from the prior episode** (no-op on first reset / when `traffic=False`), (4) call `spawner.initialize()` to spawn the fresh 20-obstacle population, (5) cache `self._last_snapshot` and `self._last_sha256`, (6) defensive flag re-clear (`arrive_flag = collision_flag = False`), (7) extract lidar and build `EpisodeInfo`; in `arena.step()` call `spawner.step()` BEFORE `env.step([action])` when traffic enabled; thread `dynamic_obstacle_count` and new `dynamic_obstacles_sha256` field through `EpisodeInfo`; update `initial_dynamic_snapshot` to return `self._last_snapshot` (or `()` when traffic=False OR pre-first-reset); call `spawner.close()` in `Arena.close()`. Add the new `dynamic_obstacles_sha256: str \| None` field to `EpisodeInfo`. Update the `EXPECTED_EPISODE_INFO_FIELDS` constant at module scope and TC2's field-list assertion accordingly. Satisfies AC1, AC5, AC13. | T1 | high | `arena/arena.py` | Two existing TCs (TC2: field-list check; TC7: counters-after-reset) will fail without the `EXPECTED_EPISODE_INFO_FIELDS` update — fix them in this task, not in T4. TC2b's monkeypatch path still works (lidar status check is independent of traffic). Be careful: `EpisodeInfo` in Phase 0 had `dynamic_obstacle_count: int` already at 0; Phase 2 keeps it `int` (just stops being always-0). **Render-mode despawn**: `env.delete_object(id)` calls `obj.plot_clear()` (per `env_base.py:956`); no extra render code, but T6's render smoke must verify no matplotlib artist leak after ~100 despawn/spawn cycles. |
| T3 | **Runner CLI: add `--traffic / --no-traffic` flag, default `--traffic`; write `dynamic_obstacles_sha256` to trace JSONL conditionally; partition outputs by world-stem**: in `runners/run_episode.py`, add a mutually-exclusive argparse group with `--traffic` (`store_true`) and `--no-traffic` (`store_false`), `default=True` for `traffic`. Pass to `Arena(..., traffic=args.traffic)`. In `_trace_line`, accept a new `dynamic_obstacles_sha256: str \| None` kwarg; when not None, include the key in the JSON dict (sort_keys handles ordering). In the run loop, source the value from `info.dynamic_obstacles_sha256`. Update the existing planner-failure branch (no trace written) and step-0 sentinel branch to pass `info0.dynamic_obstacles_sha256` — which is the reset-time hash when traffic is on, None otherwise. **World-stem partitioning**: change `out_dir = Path(args.results_dir) / args.algorithm` to `out_dir = Path(args.results_dir) / Path(args.world).stem / args.algorithm`. Update the module docstring's `Outputs:` block and the `--results-dir` argparse `help=` string to reflect `<results-dir>/<world_stem>/<algorithm>/<seed>.{json,trace.jsonl}`. Satisfies AC6, AC7, AC15. | T2 | high | `runners/run_episode.py` | The step-0 trace line currently has `action=[0.0,0.0]`. With traffic on, the step-0 line's `dynamic_obstacles_sha256` IS the initial-snapshot hash (reset-time), NOT None — TC15 / TC20 byte-determinism depend on this. Get it wrong and the symptom is "every trace line after step 0 diverges in inscrutable ways." Risk bumped to **high** (was med in v1 of this spec) because this single line is the most-fragile determinism guarantee in Phase 2. **World-stem partitioning is path-only**: no behavioral change to metrics/trace contents; only the output directory layout shifts. Path-construction uses `Path(args.world).stem` (the basename without extension), NOT the full path — so `tests/foo/arena_v1.yaml` and `arena/arena_v1.yaml` both partition into `arena_v1/`. If two YAMLs with the same stem live in different directories, they collide; this is acceptable (and matches how Phase 5 plot.py will identify worlds). |
| T4 | **Update Phase 1 TC14/TC15/TC16 path expectations to the new layout; pass `--no-traffic` on TC14/TC15; add TC17–TC21 + TC22 + TC23**: In `arena/arena.py`: (a) modify TC14's subprocess invocation to add `--no-traffic` AND update the expected JSON / JSONL paths from `Path(td) / "a_star_once" / "42.{json,trace.jsonl}"` to `Path(td) / "arena_v1" / "a_star_once" / "42.{json,trace.jsonl}"`; (b) same path update for TC15's two-subprocess paths; (c) update TC16's expected paths from `Path(td) / "a_star_once" / "0.{json}"` and `... / "0.trace.jsonl"` to `Path(td) / "arena_no_path" / "a_star_once" / "0.{json,trace.jsonl}"`. TC14's per-line key check stays at the existing 7-key strict-set assertion (TC14 runs `--no-traffic`). Add TC17, TC18, TC19, TC20, TC21 per the Testing Strategy table; register all in `_run_checks`. Add TC22 (world-stem partitioning end-to-end): subprocess-invoke the runner twice with the same seed but different `--world` (e.g. `arena_v1.yaml` and `arena_v2_hard.yaml`) into the same tempdir, assert both `<world_stem>/a_star_once/<seed>.json` files exist and have distinct contents (verifies AC15). Add TC23 (import-cycle guard): subprocess `python -c "import planners; import arena.arena"` and reversed; exit 0 on both. Total `--check` expectation: **24 PASS** (17 existing + TC17 + TC18 + TC19 + TC20 + TC21 + TC22 + TC23). **Wallclock budget**: total `--check` runtime must stay under 120 s on the dev laptop — TC22 adds two new subprocess invocations (~10-20 s each); still comfortably under budget. Satisfies AC8 (revised to 24 PASS), AC9, AC10, AC11, AC12, AC13, AC15, partial AC6/AC7. | T3 | med | `arena/arena.py` | TC19 ALWAYS uses `spawner._inject_for_test` — do NOT branch on whether the natural snapshot happens to have a near-trajectory obstacle. Determinism + reproducibility > test "naturalness." A separate informational assert in TC19 verifies the natural-traffic crash path is reachable on `seed=2`, but does not gate the test. TC22's "distinct contents" assertion can be as cheap as `json_a != json_b` after both loads — different worlds + same seed must yield different `path_length` and `time_to_goal` at minimum. |
| T5 | **Narrow `PathPlanner` Protocol param type + docs**: in `planners/_types.py`, change `initial_dynamic_snapshot: tuple` to `initial_dynamic_snapshot: tuple[DynamicObstacleState, ...]` (import `DynamicObstacleState` from `arena.dynamic`). `AStarOncePlanner` continues to ignore the param (existing `noqa: ARG002`). Add a "Phase 2 — Dynamic obstacles" section to `CLAUDE.md`: `traffic=True` constructor flag, `dynamic.py` API surface (DynamicObstacleState fields + TrafficSpawner public methods), runner default change to `--traffic`, `dynamic_obstacles_sha256` trace key, new `<results-dir>/<world_stem>/<algorithm>/<seed>` layout, one-liners for TC17–TC23, command examples. **Also update the existing "## The episode runner (Phase 1)" section** to reflect the new path layout (its "Results layout" bullets currently say `results/<algorithm>/<seed>.{json,trace.jsonl}` — change to `results/<world_stem>/<algorithm>/<seed>.{json,trace.jsonl}` and update the example invocation paths in the Run-command block). ~20-30 lines of new content, plus the Phase 1 section path tweak. Satisfies AC14. | T2 | low | `planners/_types.py`, `CLAUDE.md` | The Protocol import line is `from arena.dynamic import DynamicObstacleState`. This creates a one-way import edge planners → arena, which is fine (arena has no reverse dependency on planners). The Phase 1 section update is small (a couple of path strings) but easy to forget — flag it in PR review. |
| T6 | **Manual verification**: activate venv, run `python arena/arena.py arena/arena_v1.yaml --check` — expect **24 PASS** in <120 s wallclock. Run `python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml` (default --traffic) — expect exit 0; eyeball that the metrics JSON has the 7-key schema and the trace JSONL has 8 keys per line (including `dynamic_obstacles_sha256`). Run with `--no-traffic` — expect 7-key trace lines, byte-identical to another same-seed `--no-traffic` run. Run a short python one-liner that does `Arena('arena/arena_v1.yaml', seed=7, traffic=True, render=True); for _ in range(100): step(zero)` — visually confirm circles drift across the arena, despawn at edges, ~20 alive at any time, no leaked matplotlib artists (no stale circles staying after despawn). **Seed-flakiness escape**: if any TC fails on its declared seed AND the failure is "by design statistical" (e.g., ACs that need an obstacle within K meters), file the failure back to T1/T4 to pin a different verified seed in the AC text. Do NOT silently relax the AC. | T4, T5 | low | (observation only) | This is the gate for declaring Phase 2 done. The render check is qualitative; if you see circles spawning on edges, moving inward, and disappearing at the opposite edge with roughly 20 alive at any time, ship it. Record the AC4 lidar-byte-identity outcome (passed or fell back) in the PR description so future maintainers see which guarantee actually holds. |

## Notes for Implementer

- **`TrafficSpawner.state_sha256()` is determinism-critical.** Sort by `id` before building the (N, 5) array. Use `float64` everywhere (mixing dtypes changes the byte representation). Hash via `hashlib.sha256(arr.tobytes()).hexdigest()`.
- **traffic_rng draw order is FIXED.** Per spawn attempt: (1) perimeter position via one `uniform(0, 4*W)` draw, (2) heading via one `uniform(half_cone_lo, half_cone_hi)` draw, (3) speed via one `uniform(0.3, 1.5)` draw. On overlap rejection, repeat (1)-(3) — do NOT skip a coordinate to "reuse" the heading. Keep all draws inside one helper so reordering is a one-line code change visible in PR diffs.
- **Obstacle ids vs. monotonic indices**: irsim assigns an `obj.id` at construction; the spawner records that id in `DynamicObstacleState`. The spawner's `_next_idx` is a separate counter used ONLY for unique naming (`DYNAMIC_OBSTACLE_NAME_FMT`). Conflating the two will create name collisions after a despawn-respawn cycle.
- **Despawn during a step is safe** because irsim's collision check runs AFTER `env.step([action])`. The spawner's pre-step despawn removes only obstacles that have ALREADY exited the arena based on last-tick state — a freshly-arrived obstacle won't be despawned in the same tick it spawned in because spawn places it on the edge with an inward heading.
- **TC14b decision** (additive): if you find TC14's "8-key trace under --traffic" check fits naturally into TC14's existing structure, fold it in and keep the count at 22. If it gets messy, add TC14b as a separate case and update AC8 to "23 PASS." The spec authorizes either.
- **EpisodeInfo field-list invariant**: TC2's `EXPECTED_EPISODE_INFO_FIELDS` constant lists every field in order. Phase 2 adds `dynamic_obstacles_sha256` at the END (after `lidar_status`). Update the tuple literal in lockstep with the dataclass definition.
- **Reset re-derives rngs from master_seed** (existing Phase 0 behavior). The spawner's `initialize()` is called AFTER the rng re-derivation so traffic_rng is in the same state as a fresh `__init__` would be — guaranteeing reset() → initialize() produces the same population as construction → reset() on the same seed.
- **Runner step-0 sentinel**: with traffic on, the step-0 trace line's `dynamic_obstacles_sha256` is the reset-time hash, not None. This is critical for TC15-style byte-determinism: differ here, and every line after is meaningless to compare.
- **Static-obstacle overlap check uses inflated radius**: `point_to_obstacle_distance(spawn_xy, static_obs) < OBSTACLE_RADIUS + SPAWN_OVERLAP_BUFFER`. Do NOT use `manual_astar.SAFETY_MARGIN` here — that's the planner's margin, not the spawner's.
- **TC19 fallback API `_inject_for_test`** lives on `TrafficSpawner` as an explicitly underscore-prefixed test helper. It accepts an absolute position+velocity and bypasses the perimeter sampler, but otherwise follows the same `env.create_obstacle + env.add_object` path. It does NOT consume `traffic_rng` (so it cannot perturb determinism for tests that follow).
- **Rollback**: delete `arena/dynamic.py`; revert `arena/arena.py` to its Phase 1 state (`git diff` will show the `traffic`/spawner additions and the EpisodeInfo extension); revert `runners/run_episode.py` and `planners/_types.py`; revert `CLAUDE.md`; delete the `phase2-dynamic-obstacles` branch (T0). Repo is back to 17 PASS on `main`.
- **What this plan deliberately does NOT do**: implement any dynamic-aware planner (Phase 6); add the 50-seed master list (Phase 3 — easy to fold in here, but optional; spec does not block on it); change Mission.md's Phase 4 metrics list (no new metrics field). If the executor finds themselves writing replan logic or editing Mission.md beyond the doc-link sentence, stop — that's a different plan.

## Verified Assumptions (irsim API)

**Verified 2026-05-27 against irsim 2.9.3 source under `.venv/Lib/site-packages/irsim/`:**

- **`env.create_obstacle(kinematics={...}, shape={...}, state=[...], **kwargs)`** exists at `env/env_base.py:878` and delegates to `ObjectFactory.create_obstacle` at `world/object_factory.py:196`. With `kinematics={'name': 'omni'}` it returns an `ObjectBase` whose state can be mutated each tick (the deprecated `ObstacleOmni` class at `world/obstacles/obstacle_omni.py` confirms `kinematics={'name': 'omni'}` is the modern, supported config).
- **`env.add_object(obj)`** exists at `env/env_base.py:906`, enforces unique-name invariant, attaches the object to the env's `_objects` list, inits its plot (when `disable_all_plot=False`), and calls `build_tree()` so the spatial index is updated for the SAME tick.
- **`env.delete_object(target_id)`** exists at `env/env_base.py:946`; despawn path is supported; calls `plot_clear()` on the removed object (per `env_base.py:956`) — render-mode artists are cleaned up automatically.
- **Lidar sees newly-added obstacles on the SAME tick.** Verified at `world/sensors/lidar2d.py:240-242` in `Lidar2D.laser_geometry_process`: the lidar reads from `self._env_param.GeometryTree` and `self._env_param.objects` per query — both are the env's live state, refreshed by `build_tree()` in `add_object` / `delete_object`. No lidar-side caching at construction or reset; dynamic obstacles spawned mid-step are visible the moment the lidar runs.
- **`env.step_time`** delegates to `world.step_time` (Phase 0 T0 verified). The spawner caches `dt = env.step_time` once at construction — does NOT rely on any specific default value, only on the live attribute.
- **`render=True`** path: `env.add_object` calls `obj._init_plot(env._env_plot.ax)` and `obj._step_plot()` per `env_base.py:917-918`, so dynamic obstacles render immediately on spawn; `env.delete_object` calls `plot_clear()` so they disappear on despawn. No new render code needed.
- **`env.reset()` does NOT remove dynamically-added obstacles.** `_reset_all` (env_base.py:725-726) iterates `self.objects` and calls `obj.reset()` on each, resetting to `_init_state` but never removing. Spec consequence: `Arena.reset()` MUST delete spawner-tracked obstacles BEFORE calling `spawner.initialize()`, else population doubles per episode. See Decisions § "Reset must delete previously-spawned dynamic obstacles BEFORE re-spawning" and T2's stepwise ordering.
