# Path Planning Arena

A controlled comparison of path-planning algorithms on a shared, reproducible
arena with crossing traffic, built on top of [`irsim`](https://github.com/hanruihua/ir-sim)
(a 2D differential-drive robot simulator).

The end product is a 2D scatter plot of **time-to-goal vs. crash-rate** —
down-and-left wins. The deeper goal is to understand *why* some planners are
fast, some are safe, and whether the two properties trade off. The full
research design lives in [`Mission.md`](Mission.md).

Every algorithm runs against the same 50 seeded traffic streams in the same
50×50 world, start `(2, 2)` → goal `(48, 48)`, so the cross-algorithm
comparison is apples-to-apples.

---

## Repository layout

```
pathplanning/
├── arena/                  # the seeded test environment (Phase 0 + 2)
│   ├── arena.py            #   Arena: wraps irsim, uniform step() API, --check suite
│   ├── dynamic.py          #   DynamicObstacle + TrafficSpawner (crossing traffic)
│   ├── arena_v1.yaml       #   canonical 50×50 world (walls + 12 pillars)
│   ├── arena_v2_hard.yaml  #   second 50×50 world (walls relocated)
│   └── arena_no_path.yaml  #   start boxed in → A* must fail (failure-path fixture)
├── planners/               # pluggable planner adapters (Phase 6)
│   ├── _types.py           #   Controller protocol (reset + act) + Path type
│   ├── _grid.py            #   shared grid substrate + lidar fold + ALGORITHMS registry
│   ├── a_star.py           #   a_star_once / a_star_replan
│   ├── dijkstra.py         #   dijkstra_once / dijkstra_replan (A* with a zero heuristic)
│   └── d_star_lite.py      #   d_star_lite (incremental; rejects --replan-k)
├── runners/                # experiment harness (Phase 1 + 3)
│   ├── run_episode.py      #   one planner × one seed × one world → metrics + trace
│   └── run_experiment.py   #   one planner × the canonical 50 seeds → batch + manifest
├── results/                # generated metrics/traces (gitignored except .gitkeep)
├── docs/plans/             # per-phase implementation plans
├── manual.py               # standalone demo: naive go-to-goal
├── manual_obstacle.py      # standalone demo: reactive lidar avoidance
├── manual_astar.py         # standalone demo: A* planner + waypoint follower
├── test.py                 # standalone demo: minimal irsim "hello world"
├── *.yaml                  # demo worlds (robot_world, obstacle, obstacle_harder)
├── tests/                  # A* edge-case world fixtures (inputs, not pytest files)
├── Mission.md              # the research plan (phases 0–7)
└── requirements.txt        # irsim, numpy, pyyaml
```

The single-file demos (`test.py`, `manual*.py`) are self-contained and don't
share code with each other. The `arena/` + `planners/` + `runners/` stack is
the reusable harness that drives the actual comparison study.

---

## Setup

Windows + PowerShell. A `.venv/` is already provisioned at the repo root.

```powershell
# Activate the virtual environment (do this in every new shell)
.venv\Scripts\Activate.ps1

# Install / refresh dependencies if needed
pip install -r requirements.txt
```

Dependencies: `irsim`, `numpy`, `pyyaml`. There is no separate build step.

---

## Quick start

```powershell
.venv\Scripts\Activate.ps1

# 1. Eyeball the canonical world in a render window
python arena/arena.py arena/arena_v1.yaml --render

# 2. Run A* against one seed (traffic on by default)
python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml

# 3. Run A* against all 50 canonical seeds
python -m runners.run_experiment --algorithm a_star_once --world arena/arena_v1.yaml

# Results land under results/arena_v1/a_star_once/
```

---

## The standalone demos

Each opens an irsim render window. Run them directly with `python`.

| Command | What it does |
| --- | --- |
| `python test.py` | Minimal irsim "hello world" on `robot_world.yaml`. |
| `python manual.py` | Pure proportional go-to-goal on `obstacle.yaml`. No obstacle awareness. |
| `python manual_obstacle.py` | Reactive lidar avoidance on `obstacle_harder.yaml`. Repulsive turn from close returns + a left/right clearance bias. |
| `python manual_astar.py` | A* global planner + waypoint follower on `obstacle_harder.yaml`. |
| `python manual_astar.py tests\no_path.yaml` | Run the A* planner against a specific world (positional arg). |

`manual_astar.py` is the substantive demo: it parses the world YAML into a
frozen `WorldModel`, rasterizes an occupancy grid inflated by the robot radius
plus a safety margin, runs 8-connected A* with no corner-cutting, collapses the
dense grid path into a small set of line-of-sight-checked waypoints, then
follows them with a heading-gated speed schedule. All tuning knobs are the
`UPPER_SNAKE_CASE` constants at the top of the file.

---

## The experiment harness

Three layers, each usable from the command line.

### 1. Arena — the seeded environment

`arena/arena.py` wraps irsim and exposes a uniform
`step(action) -> (state, lidar, done, info)` interface. The canonical world is
`arena/arena_v1.yaml` (50×50, two staggered length-30 walls + 12 circle
pillars). Pass `traffic=True` to spawn a continuously refilled ~20-obstacle
population of straight-line crossing traffic.

```powershell
# Visible smoke loop — drive the world and watch the render window
python arena/arena.py arena/arena_v1.yaml --render

# Headless verification suite (38 checks, TC1–TC37; ~30 min)
python arena/arena.py arena/arena_v1.yaml --check
```

`--check` is the health gate for the whole harness. It covers the Arena API,
the episode runner, the traffic substrate, the batch runner, and the planner
family end-to-end. All 38 PASS means the harness is healthy. (With neither flag,
it defaults to `--check`.)

| Flag | Default | Meaning |
| --- | --- | --- |
| `yaml_path` (positional) | required | World YAML, e.g. `arena/arena_v1.yaml`. |
| `--seed N` | 42 | Master seed for the smoke/check run. |
| `--render` | off | Interactive smoke loop in a visible window. |
| `--check` | (default) | Run the headless TC1–TC37 verification suite. |

### 2. `run_episode` — one planner, one seed

`runners/run_episode.py` wires a registered planner to the Arena, runs a single
episode, and writes per-episode metrics plus a step-by-step trace.

```powershell
python -m runners.run_episode --algorithm a_star_once --seed 42 --world arena/arena_v1.yaml
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--algorithm NAME` | required | Registered planner, e.g. `a_star_once`. |
| `--seed N` | required | Episode seed. |
| `--world PATH` | required | World YAML. |
| `--replan-k N` | none | Replan cadence; required for the `_replan` family, forbidden otherwise. |
| `--render` | off | Open the irsim render window. |
| `--results-dir DIR` | `results` | Override the output directory. |
| `--traffic` / `--no-traffic` | traffic on | Toggle Phase 2 crossing traffic. |

A* `_once` planners don't dodge, so most traffic seeds end in collision — that
is the experimental signal the scatter plot consumes. Use `--no-traffic` to
reproduce the deterministic static-world success path.

### 3. `run_experiment` — one planner, the canonical 50 seeds

`runners/run_experiment.py` derives 50 seeds from a single master seed via
`SeedSequence.spawn` and shells out to `run_episode` once per seed (one fresh
irsim subprocess each). This is what guarantees every algorithm faces the same
50 traffic streams.

```powershell
python -m runners.run_experiment --algorithm a_star_once --world arena/arena_v1.yaml
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--algorithm NAME` | required | Registered planner. |
| `--world PATH` | required | World YAML. |
| `--replan-k N` | none | Replan cadence; required for the `_replan` family, forbidden otherwise. Forwarded to each episode and recorded in the manifest. |
| `--master-seed N` | 20260605 | Master seed the 50 episode seeds derive from. |
| `--num-seeds N` | 50 | Run a prefix of the canonical stream (prefix-stable). |
| `--jobs N` | 1 | `1` = sequential. `N>1` = up to N concurrent subprocesses. |
| `--results-dir DIR` | `results` | Forwarded to each episode. |
| `--resume` | off | Skip seeds whose `<seed>.json` already exists. |
| `--traffic` / `--no-traffic` | traffic on | Forwarded to each episode. |

Result bytes are identical at any `--jobs` value; only `wallclock_per_step`
(a Mission.md "freebie" metric) is perturbed by contention. Produce headline
wall-clock numbers with `--jobs 1`.

---

## Results layout

Output is partitioned by world stem so the same seed against two different
worlds never clobbers itself:

```
results/<world_stem>/<algorithm>/
├── <seed>.json          # 7-field metrics, one object per episode
├── <seed>.trace.jsonl   # per-step trace (only written on planning success)
└── _manifest.json       # provenance receipt (run_experiment only)
```

`<world_stem>` is `Path(--world).stem`, so `arena/arena_v1.yaml` →
`results/arena_v1/`. `results/` is gitignored except for `.gitkeep`.

**Metrics JSON** (`<seed>.json`) — 7 fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `time_to_goal` | `float \| null` | Sim seconds to reach the goal; null on crash, timeout, or planner error. |
| `crashed` | `bool` | irsim collision flag. |
| `timed_out` | `bool` | `sim_time >= 120.0` without reaching the goal. |
| `path_length` | `float` | Σ of per-step XY displacement over the executed trajectory. |
| `mean_speed` | `float` | `path_length / sim_time`. |
| `wallclock_per_step` | `float` | Mean wall-clock per step (a `perf_counter` mean; not byte-deterministic). |
| `planner_error` | `str \| null` | Exception message if the t=0 plan in `reset()` raised, else null. |

**Trace JSONL** (`<seed>.trace.jsonl`) — one JSON object per line, keys sorted:
`step`, `state` `[x, y, θ]`, `action` `[v, ω]`, `lidar_sha256`, `crashed`,
`reached_goal`, `done`. With traffic on, an 8th key
`dynamic_obstacles_sha256` is added per line. Step 0 records the post-reset
state with a sentinel `action=[0.0, 0.0]`.

**Manifest** (`_manifest.json`) — `master_seed`, `num_seeds`,
`derived_seeds`, per-episode `{seed, exit_code, status}` in derivation order,
and a best-effort `git_sha`. No timestamps, so it is byte-reproducible.

---

## Determinism

The harness is built so the same seed always produces the same bytes:

- Same seed → **byte-identical** `<seed>.trace.jsonl` across runs.
- Two same-master-seed `run_experiment` runs → byte-identical per-seed JSON
  and `_manifest.json`.
- A `--jobs N` run keeps the manifest in derivation order (completion order
  never leaks into the output).

The one exception is `wallclock_per_step`, a real-time `perf_counter` mean that
cannot be byte-identical across two live runs.

Traffic substreams are derived from the master seed via
`SeedSequence.spawn(2)` (`traffic_rng` for spawning, `motion_rng` reserved for
future motion noise), drawn in a fixed order per spawn attempt.

---

## Adding a planner

Planners live in `planners/<name>.py` and satisfy the `Controller` protocol in
`planners/_types.py`:

```python
class Controller(Protocol):
    name: str  # the FAMILY name, e.g. "a_star_replan"; the results label adds _k<K>

    def reset(self, world_yaml, initial_snapshot, lidar0, state0) -> None: ...
    def act(self, state, lidar) -> np.ndarray: ...  # (2,1) float [[v],[w]]
```

`reset()` builds the static substrate and the t=0 plan (raise `ValueError` /
`RuntimeError` to surface a no-path as `planner_error`); `act()` returns the next
`(2,1)` action and must not raise on a mid-episode replan failure. The runner
calls `reset()` once, then `act()` until the Arena reports done.

Register the class by self-registering into the `ALGORITHMS` registry: the
controller module calls `register(name, cls)` from `planners/_grid.py` at import
(see `a_star.py`), and importing the `planners` package populates the registry.
The runner builds the instance via `build_controller`. Five planners ship today:
`a_star_once`, `a_star_replan`, `dijkstra_once`, `dijkstra_replan`, and
`d_star_lite`. The `_replan` families take a required `--replan-k`; `d_star_lite`
is the incremental planner (no `_once`/`_replan` split, and it rejects
`--replan-k`). Mission.md Phase 6 still expects the reactive (DWA, APF) and
sampling (RRT, RRT*) families.

---

## World YAML schema

All scripts consume the irsim world format. The fields the scripts rely on:

- `world.width`, `world.height`, optional `world.offset` (sizes the occupancy grid)
- `robot.shape.radius` (obstacles are inflated by this + a safety margin)
- `robot.state` = `[x, y, theta]` start pose; `robot.goal` = `[x, y, theta]` goal pose
- `robot.sensors` — a `lidar2d` entry (required by `manual_obstacle.py` and by `Arena`)
- `obstacle[]` with `shape.name` in `{circle, rectangle, polygon, linestring}`

irsim is strict about field shapes — when adding a world, copy an existing one
as the template. World fixtures that live in the repo go in `tests/`; scratch
worlds belong outside the repo or under the gitignored `_tmp_*` prefix.

---

## Project status

Following the phase plan in `Mission.md`:

| Phase | Status | Deliverable |
| --- | --- | --- |
| 0 — Arena | done | `arena/arena.py` + `arena_v1.yaml` |
| 1 — Harness sanity check | done | `runners/run_episode.py` + metrics/trace |
| 2 — Dynamic obstacles | done | `arena/dynamic.py` crossing traffic |
| 3 — Reproducibility | done | `runners/run_experiment.py` + manifest |
| 4 — Metrics | pending | per-algorithm aggregation |
| 5 — Scatter plot | pending | `results/plot.py` |
| 6 — Algorithms | in progress | `planners/` (Controller interface + grid family A*/Dijkstra once+replan + D* Lite landed; reactive DWA/APF, sampling RRT/RRT*, and the 6b K-sweep remain) |
| 7 — The actual question | pending | the insight the plot produces |

Phase-by-phase implementation notes live in `docs/plans/`. Per-phase
architecture and conventions are documented in `CLAUDE.md`.
