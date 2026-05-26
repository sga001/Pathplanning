# Mission

A controlled comparison of path-planning algorithms on a shared, reproducible
arena with crossing traffic. The output is a 2D scatter plot of
**time-to-goal vs. crash-rate** — down-and-left wins. The deeper goal is to
understand *why* some planners are fast, some are safe, and whether the two
properties trade off.

---

## Locked-in decisions

| Decision | Value |
| --- | --- |
| Arena size | **50 × 50** meters |
| Robot model | irsim differential-drive (reuse from existing scripts) |
| Static layout | **Walls + pillars mix** (a couple of long walls forcing corridors, plus 10–20 scattered pillars) |
| Start / goal | **Fixed**: `(2, 2) → (48, 48)` (opposite corners), all seeds |
| Goal-reached detection | Use irsim's built-in goal detection |
| Crash detection | Use irsim's built-in collision flag |
| Dynamic obstacle model | **Spawned + linear** crossing traffic: random heading/speed at the edges, straight-line motion, despawn on exit |
| **Population is maintained at ~20** | New obstacles respawn as soon as old ones exit. Robot can't game the experiment by waiting for traffic to clear. |
| Dynamic obstacle speed | **0.3 – 1.5 × robot top speed**, sampled per obstacle |
| Dynamic obstacle shape | Uniform small circles, **r = 0.3 m** |
| Robot observation | **Lidar only** for the run-time stream |
| Planner prior knowledge (t = 0) | **Static map + initial positions of all dynamic obstacles currently in the arena**. After t = 0, only lidar. |
| Episode timeout | **120 s of sim time** |
| Timeouts | **Folded into crash rate** (counted as failures) |
| Compute budget | **None** — sim-time-to-goal only; log wall-clock as a freebie |
| Seeds per algorithm | **50** (every algorithm runs against the same 50 traffic streams) |
| Implementation policy | Libraries where maintained ones exist; otherwise hand-roll in `manual_astar.py` style |

Algorithms in scope: **A\***, **Dijkstra**, **D\* Lite**, **DWA**, **APF**
("push-away" renamed), **RRT**, **RRT\***. GMapping dropped (it's SLAM).

---

## Phase 0 — Arena

A reusable test environment that every algorithm runs against on identical
(seeded) conditions.

- 50 × 50 arena; static walls + pillars in YAML; diff-drive robot.
- Fixed start `(2, 2)`, goal `(48, 48)`.
- New `arena/` directory; canonical world `arena/arena_v1.yaml`.
- Python entry point `arena.py` that wraps irsim and exposes a uniform
  `step(action) -> (state, lidar, done, info)` interface.
- The `arena` module is responsible for spawning the dynamic-obstacle
  population from a seeded RNG; planners do not see this code path.

Open questions:

- Concrete static layout: how many pillars, where do the walls go? Sketch
  first, then encode in YAML.

---

## Phase 1 — Harness sanity check

Run A* inside the harness against the static-only world (no dynamic
obstacles yet) and confirm:

- A single episode runs start → goal end-to-end.
- irsim's collision flag fires when we deliberately drive into a wall.
- irsim's goal detection fires when we reach `(48, 48)`.
- Per-episode metrics get logged to `results/<algorithm>/<seed>.json`:
  `{ time_to_goal, crashed, timed_out, path_length, mean_speed,
  wallclock_per_step, planner_error }`.
- Same seed → byte-identical trace.

---

## Phase 2 — Dynamic obstacles (crossing traffic)

Crossing-traffic model: obstacles spawn at the arena edges with random
heading and speed, travel straight, despawn when they exit. Population
maintained at **~20** by respawning immediately.

- `DynamicObstacle` class with `step(dt, rng)` advancing position.
- `TrafficSpawner` maintains the target population and uses the seeded RNG
  for spawn position / heading / speed.
- Obstacles pass through each other.
- Each obstacle is rendered into the irsim world so the lidar sees it
  naturally — no special observation channel.

Critical design constraint: **population must be continuously refilled**.
Otherwise the robot can wait out the traffic, which collapses the
experiment.

Open questions:

- Spawn-point distribution along the edges (uniform? edge-biased?) and
  heading distribution (uniform over outward-pointing? biased toward
  arena center?). Default to whatever is simplest and reasonable; revisit
  if the traffic feels degenerate.

---

## Phase 3 — Reproducibility

- One master seed per experiment; sub-streams for traffic spawning and
  motion derived from it (via `numpy.random.SeedSequence`).
- Reproducibility test as part of Phase 1: run seed K twice, diff JSONL —
  must be identical.
- All algorithms run against the **same 50 seeds**, so the same traffic
  stream challenges each planner. This is what makes the cross-algorithm
  scatter plot meaningful.

---

## Phase 4 — Metrics

Per episode:

1. **Time to goal** (sim seconds; undefined for crashes/timeouts).
2. **Crashed** (irsim collision flag).
3. **Timed out** (hit 120 s without reaching goal — folded into crash rate
   as a failure).
4. Bonus telemetry: path length, mean speed, mean wall-clock per planning
   step.

Aggregated per algorithm (over 50 seeds):

- Time-to-goal distribution (successes only).
- Failure rate = `(crashes + timeouts) / 50`.

---

## Phase 5 — The scatter plot

- X = time to goal. Y = failure rate (crash + timeout).
- **Every per-seed run plotted as a dot** so the full distribution is
  visible per algorithm.
- **Both median and mean shown as centroid markers** per algorithm, so the
  reader can see whether the tail is dragging the mean.
- One color per algorithm; legend on the side.
- matplotlib, static PNG, generated by `results/plot.py` reading
  `results/<algorithm>/*.json`.

---

## Phase 6 — Algorithms

Each planner lives in `planners/<name>.py`. Interface:

- **Path planners** output a `Path` (list of waypoints). A shared
  `WaypointFollower` (reused from `manual_astar.py`) executes them.
- **Reactive planners** output `(v, ω)` directly.
- The harness dispatches on output type.

### Algorithm list and variations

**A\*** and **Dijkstra** and **RRT** and **RRT\*** each get **two
variants** — this is a deliberate experiment:

- `*_once`: plan a single path at t = 0 (using static map + initial
  obstacle snapshot), then follow it forever. Tests "how brittle is a
  static plan in dynamic traffic?"
- `*_replan_K`: re-plan every **K** sim steps using the latest lidar-built
  obstacle layer. Tests "does periodic replanning recover the gap to
  incremental planners?"

The value of **K** is itself a research question (see Phase 6b).

- **A\*** (`a_star_once`, `a_star_replan_K`) — grid + heuristic. Baseline.
- **Dijkstra** (`dijkstra_once`, `dijkstra_replan_K`) — grid, no
  heuristic. Sanity-check that the heuristic is doing work.
- **D\* Lite** — *incremental* replanner; no `_once` / `_replan_K` split
  because it updates only the affected portion of the search graph as the
  world changes. Hand-rolled in `manual_astar.py` style.
- **DWA** — reactive, samples feasible `(v, ω)` over a short horizon.
  Velocity output. Hand-rolled.
- **APF** — Artificial Potential Fields (Khatib 1986); attractive force
  to goal + repulsive force from lidar-detected obstacles. Velocity
  output. Hand-rolled.
- **RRT** (`rrt_once`, `rrt_replan_K`) — sampling-based, fast/sloppy.
  Library if `pyrrt` or similar is maintained; otherwise hand-rolled.
- **RRT\*** (`rrt_star_once`, `rrt_star_replan_K`) — asymptotically
  optimal sampling planner. Same library/hand-roll policy as RRT.

### Phase 6b — Finding the right K

For each `*_replan_K` planner, sweep K over (e.g.) `{1, 5, 25, 100, ∞}`
sim steps. `K = ∞` is equivalent to `*_once`. Plot K vs. failure rate +
time-to-goal **per planner**.

Hypotheses to test:

- Failure rate falls monotonically as K decreases (more replanning =
  safer). True for sampling planners? For grid planners?
- Time-to-goal has a sweet spot in K — too frequent and you jitter, too
  rare and you collide.
- The optimal K for A* and for RRT* differ in a principled way (grid
  planners benefit less from constant replanning because their solution
  is already deterministic given the map).

The best K per planner gets picked for that planner's headline dot on the
main scatter plot.

---

## Phase 7 — The actual question

Once the plot exists:

- **What makes a planner fast?** Hypothesis: global knowledge + good
  heuristic. Test: A* (with heuristic) vs. Dijkstra (without), at matched
  K.
- **What makes a planner safe?** Hypothesis: high-rate local reactivity +
  short reaction horizon. Test: DWA vs. A*_replan_K at low K.
- **Does incremental replanning (D\* Lite) dominate periodic replanning
  at the right K?** If yes, the algorithmic property of incrementality
  matters; if no, periodic-but-fast is good enough.
- **Is there a Pareto frontier, or do D\* Lite / RRT\*_replan land alone
  in the down-left corner?** If they do, the "tension" between fast and
  safe is more myth than law.
- **How does the Pareto frontier change with traffic density?** (Stretch
  goal — would need to re-run at 5 / 10 / 20 obstacles.)

This is the only phase that produces an *insight* rather than an
artifact. Everything before it exists to make this comparison rigorous.

---

## Still TBD

- Concrete static layout (wall positions + pillar count/positions).
- Spawn-point and heading distributions for dynamic obstacles.
- K-sweep values for the replan experiment.
- Whether `pyrrt` (or equivalent) is maintained, or we hand-roll RRT/RRT*.
