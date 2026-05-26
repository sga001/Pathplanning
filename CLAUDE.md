# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A small sandbox of differential-drive path-planning demos built on top of [`irsim`](https://github.com/hanruihua/ir-sim) (2D robot simulator). Each top-level Python script is a self-contained controller experiment — they don't share modules with each other. World layouts live in YAML files at the repo root, A* edge-case fixtures live in `tests/`.

## Environment & commands

Windows + PowerShell. A `.venv/` is already provisioned at the repo root.

```powershell
# Activate the venv before running anything
.venv\Scripts\Activate.ps1

# Install / refresh dependencies
pip install -r requirements.txt

# Run the demos (each opens an irsim render window)
python test.py                          # minimal "irsim hello world" using robot_world.yaml
python manual.py                        # naive go-to-goal on obstacle.yaml
python manual_obstacle.py               # reactive lidar avoidance on obstacle_harder.yaml
python manual_astar.py                  # default: A* + waypoint follower on obstacle_harder.yaml
python manual_astar.py tests\no_path.yaml   # planner on a specific world (positional arg)
```

There is no test runner, linter, or build step configured. The `tests/` directory currently contains A* world fixtures (`blocked_start.yaml`, `no_path.yaml`, `partial_block.yaml`) used by hand against `manual_astar.py` — they are inputs, not pytest files.

## World YAML schema

All scripts consume the same irsim world format. The fields the scripts actually rely on:

- `world.width`, `world.height`, optional `world.offset` (planner reads these to size its occupancy grid)
- `robot.shape.radius` (planner inflates obstacles by this + a safety margin)
- `robot.state` = `[x, y, theta]` start pose
- `robot.goal` = `[x, y, theta]` goal pose
- `robot.sensors` — only `manual_obstacle.py` requires a `lidar2d` entry (see `obstacle_harder.yaml` for the canonical config)
- `obstacle[]` with `shape.name` in `{circle, rectangle, polygon, linestring}`. Polygons/linestrings can carry a `state` pose that the planner applies as a rotate+translate.

When adding a new world, copy an existing one as the template — irsim is strict about field shapes.

## The three controllers, at a glance

1. **`manual.py`** — pure proportional go-to-goal: heading error → angular velocity, constant linear velocity. No obstacle awareness; only works on `obstacle.yaml` where the start pose is already clear of the central blocker.

2. **`manual_obstacle.py`** — reactive lidar avoider. Reads `robot.get_lidar_scan()`, computes a repulsive turn from close-range returns plus a side-bias term from left-vs-right mean clearance. Single `action()` function dispatches on `closest_forward_distance` thresholds (escape / side-bias / slow / caution / cruise / turning). All tunables are module-level constants at the top of the file.

3. **`manual_astar.py`** — the substantive script. Global planner pipeline:
   - `load_world()` parses the YAML into a `WorldModel` (frozen dataclass) with normalized obstacle specs (circle / rectangle / polygon / linestring → `ObstacleSpec`).
   - `build_occupancy_grid()` rasterizes the world at `GRID_RESOLUTION` (0.1 m), marking any cell within `robot_radius + SAFETY_MARGIN` of any obstacle as blocked. Uses analytic distance per obstacle kind (`point_to_obstacle_distance`).
   - `astar_search()` runs 8-connected A* with octile-distance step cost and Euclidean heuristic; diagonal moves are blocked if either orthogonal neighbor is occupied (no corner-cutting).
   - `path_to_waypoints()` collapses the dense grid path into a sparse waypoint list by sampling at `WAYPOINT_STRIDE`, then recursively bisecting any segment that fails an inflation-aware line-of-sight check (`segment_is_clear`). This is the key non-obvious step — it turns the staircase grid path into a small set of safe waypoints.
   - `WaypointFollower` + `compute_action()` advance the waypoint index when within `WAYPOINT_REACHED_DISTANCE`, then apply a heading-gated speed schedule (full speed only when heading error is small).

   Tuning knobs are the `UPPER_SNAKE_CASE` constants at the top of the file — change those rather than threading parameters through call sites.

## The arena harness (Phase 0)

`arena/` is a reusable seeded 50×50 test environment wrapping irsim, intended as the shared substrate for every planner experiment in Mission.md. Phase 0 contains static obstacles only; dynamic traffic plugs in at Phase 2 behind the `initial_dynamic_snapshot` seam.

**API:**
- `Arena(yaml_path, seed, render=False, timeout_s=120.0)` — construct; validates lidar config at init time.
- `reset() -> (state, lidar, info)` — returns `state` as `np.ndarray` shape `(3,)` (x, y, theta), `lidar` shape `(360,)` float64 (NaN = no return), and an `EpisodeInfo` frozen dataclass.
- `step(action) -> (state, lidar, done, info)` — `action` is `np.ndarray([[v],[w]], dtype=float)` shape `(2,1)`; raises `ValueError` on bad input, `RuntimeError` if called after `done`.
- `arena.close()` — tears down the irsim env. Always call in a `finally` block.
- `arena.initial_dynamic_snapshot` — returns `()` in Phase 0; Phase 2 narrows the type.

**Smoke and verification:**
```powershell
.venv\Scripts\Activate.ps1
python arena/arena.py arena/arena_v1.yaml --check     # 13 PASS = harness healthy
python arena/arena.py arena/arena_v1.yaml --render    # visible smoke loop (use to eyeball YAML)
```

`arena/arena_v1.yaml` is the canonical world: 50×50, robot start (2,2) → goal (48,48), two staggered length-30 rectangle walls + 12 circle pillars (14 obstacles total).

Phase 2 will plug `DynamicObstacle` / `TrafficSpawner` behind `Arena.initial_dynamic_snapshot` and consume the already-plumbed `traffic_rng` / `motion_rng` from `__init__`. Do not add dynamic obstacle code to Phase 0.

## The episode runner (Phase 1)

`runners/run_episode.py` is the harness entry point that wires a planner to the Arena and records per-episode metrics and a step-by-step trace.

**Run command:**
```powershell
.venv\Scripts\Activate.ps1
python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml
```

Optional flags: `--render` (opens the irsim render window) and `--results-dir <dir>` (overrides the default `results/` output directory).

**Results layout:**
- `results/<algorithm>/<seed>.json` — per-episode metrics (one JSON object).
- `results/<algorithm>/<seed>.trace.jsonl` — per-step trace (one JSON object per line, keys sorted); written only if planning succeeded (i.e., `planner_error` is null).
- `results/` is gitignored except for `.gitkeep`.

**Metrics JSON schema** (7 fields — extends Mission.md Phase 1's original 6-field list by adding `planner_error`):
- `time_to_goal: float | null` — sim seconds to reach goal on success; null on crash, timeout, or planner error.
- `crashed: bool` — irsim collision flag.
- `timed_out: bool` — sim_time >= 120.0 without reaching goal.
- `path_length: float` — Σ ‖state[t+1][:2] − state[t][:2]‖ over the executed trajectory.
- `mean_speed: float` — path_length / sim_time.
- `wallclock_per_step: float` — mean of `EpisodeInfo.wallclock_per_step` across all steps; NOT byte-deterministic across real-time runs (perf_counter mean).
- `planner_error: str | null` — exception message if `plan()` raised, else null.

**Trace JSONL schema** (one JSON object per line, keys sorted):
- `step: int`, `state: [x, y, θ]`, `action: [v, ω]`, `lidar_sha256: str` (SHA256 hex of `lidar.tobytes()`), `crashed: bool`, `reached_goal: bool`, `done: bool`.
- Step 0 records the post-reset state with `action=[0.0, 0.0]` as a sentinel; subsequent steps record state AFTER each `arena.step(action)`.

**Determinism guarantee:** same seed → byte-identical `<seed>.trace.jsonl` files across runs. Metrics JSON is equal in every field EXCEPT `wallclock_per_step`, which is a `perf_counter` mean and cannot be byte-identical across two real-time runs.

**TC13–TC16** (added to `python arena/arena.py arena/arena_v1.yaml --check`):
- TC13: scripted wall-crash via teleport — proves irsim's `collision_flag` fires on a rectangle wall.
- TC14: full A* drive through the runner (subprocess) + trace-line schema audit — verifies all 7 trace fields are present and typed correctly.
- TC15: byte-identical trace JSONL across two seeded subprocess runs — verifies the determinism guarantee end-to-end.
- TC16: planner-failure path on `arena/arena_no_path.yaml` — verifies that a sealed-box world causes A* to raise and that `planner_error` is populated and `trace.jsonl` is not written.

**`arena/arena_no_path.yaml` fixture:** A Phase-1-only Arena-compatible world where the goal is sealed off so A* cannot find a path (used exclusively by TC16). The legacy `tests/no_path.yaml` cannot substitute here because it lacks the `lidar2d` sensor block that `Arena.__init__` requires.

## Conventions worth preserving

- `manual_astar.py` is written in a strict, dataclass-heavy style (frozen dataclasses, exhaustive `raise ValueError`s on bad input, type hints everywhere, no magic numbers in function bodies). New planner code in this file should match that style; the other scripts are deliberately looser.
- World YAML filenames spell "obstacle" correctly. The earlier "obstical" spelling was renamed — don't reintroduce it.
- Scratch worlds belong outside the repo or under the `_tmp_*` prefix (gitignored). World fixtures intended to live in the repo go in `tests/`.
