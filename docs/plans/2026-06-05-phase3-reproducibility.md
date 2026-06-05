# Phase 3 — Reproducibility (Batch Experiment Runner) Plan

**Goal:** Deliver the one piece of Mission.md's "Phase 3 — Reproducibility" that isn't already built: a **batch experiment runner** that derives 50 per-episode seeds from a single master seed via `SeedSequence` and runs them all against one (algorithm, world), so every algorithm in the study faces the *same 50 traffic streams* — the property that makes the cross-algorithm scatter plot meaningful.

**Approach:** Add `runners/run_experiment.py`, a thin orchestration layer over the already-shipping, already-deterministic single-episode runner (`runners/run_episode.py`). It computes `SeedSequence(MASTER).spawn(num_seeds)` → one 64-bit int seed per episode, then launches `python -m runners.run_episode --seed <derived> ...` once per seed. Execution is **sequential by default (`--jobs 1`)** with an opt-in **bounded-concurrency** mode (`--jobs N`, a `ThreadPoolExecutor` over `subprocess.run` — threads, not a multiprocessing pool, so Windows `spawn`/pickling never enters the picture). It overwrites by default (`--resume` to skip existing files), continues past a failed seed and reports at the end (non-zero batch exit if any seed's subprocess failed), and writes a deterministic `_manifest.json` provenance receipt. No changes to `Arena`, `planners`, or `run_episode.py` behavior; per-episode determinism (TC15/TC20/TC24) and the `SeedSequence.spawn(2)` traffic/motion substreams already exist and are reused as-is.

## Scope

- **In scope:**
  - `runners/run_experiment.py` (new) — batch CLI; `derive_episode_seeds(master, n)` helper; sequential + `--jobs N` bounded-concurrency execution over `python -m runners.run_episode` subprocesses; overwrite/`--resume`; continue-and-report failure policy; `_manifest.json` writer; a pure `summarize(...)` tally helper.
  - `arena/arena.py` — three new fast, fully-deterministic test cases registered in `_run_checks`: TC25 (seed-derivation determinism/uniqueness/prefix), TC26 (batch orchestration determinism on a fast-failing world), TC27 (failure accounting + non-zero batch exit).
  - `CLAUDE.md` — new "## The batch experiment runner (Phase 3)" section; **correct the stale "`--check` … 25 PASS … under 120 s" claim** (measured ~9–10 min on the dev laptop) in the existing arena/runner sections.
  - `Mission.md` — optional one-line note that Phase 3's batch-runner deliverable landed (no semantic edits to the phase definitions).
- **Out of scope:**
  - **Per-algorithm aggregation** (time-to-goal distribution over successes, failure rate `= (crashes+timeouts)/50`) — Mission.md **Phase 4**. The batch runner produces the 50 per-episode `*.json`; it does NOT compute summary statistics.
  - **`results/plot.py`** scatter plot — Mission.md **Phase 5**.
  - **New planners** (`a_star_replan_K`, Dijkstra, DWA, APF, D\* Lite, RRT, RRT\*) and the K-sweep — Mission.md **Phase 6 / 6b**. The registry still holds only `a_star_once`.
  - **Changing `Arena`'s seed API.** Arena keeps its `int` seed; each derived int already spawns its own `traffic_rng`/`motion_rng` substreams. No `SeedSequence`/entropy parameter is added.
  - **Changing `run_episode.py` behavior.** The batch runner shells out to the existing entry point unchanged (it may *import* `ALGORITHMS` from it for up-front validation — an import edge, not a behavior change).
  - **A pytest harness.** Tests stay as `TCi` functions under `arena/arena.py --check`, per project convention.
  - **In-process episode execution.** Rejected for global-state isolation (see Decisions); every episode is its own subprocess.

## Decisions

- **New module `runners/run_experiment.py`, separate from `run_episode.py`** — The single-episode runner stays the stable, tested unit; the batch layer composes over it via subprocess. Rejected: adding a `--batch` mode to `run_episode.py` (overloads one entry point with two responsibilities and risks regressing the determinism-critical single-episode path).
- **Seed derivation: `SeedSequence(MASTER).spawn(num_seeds)`, each child → one 64-bit int** — Literal Mission.md ("one master seed per experiment; sub-streams derived … via `numpy.random.SeedSequence`"). Each child's first two `uint32` state words are packed into a 64-bit int (`w0 | (w1 << 32)`) used as the episode `--seed`. **64-bit (not 32-bit)** so the birthday-collision probability over 50 draws is ~nil (a 32-bit collision is ~3e-7 per batch and its failure mode — two episodes sharing a `<seed>.json` filename, silently yielding 49 files — is nasty). A `derive_episode_seeds` uniqueness assertion makes even that impossible collision **loud and reproducible**. Rejected: fixed `range(50)` (doesn't match Mission.md's "derive sub-streams from one master via SeedSequence"); passing the child `SeedSequence` into `Arena` (would change Arena's API, which the user pinned as out of scope).
- **Prefix property is load-bearing and documented** — Because `SeedSequence(M).spawn(k)[i]` depends only on the child spawn-key `(i,)`, `derive_episode_seeds(M, 3) == derive_episode_seeds(M, 50)[:3]`. So the reduced-N reproducibility TC (N=3) exercises the *exact same first 3 seeds* a full N=50 run would, and `--num-seeds` is a clean prefix selector, not a different stream. TC25 asserts this.
- **Execution: sequential default (`--jobs 1`); opt-in `--jobs N` bounded concurrency via `ThreadPoolExecutor` over `subprocess.run`** — Outcome of the `/debate` on this exact decision (verdict **Lean Yes, 7.25/10**, grounded on a **measured 58.2 s/episode** → ~48 min per 50-seed batch sequentially). Threads (not `ProcessPoolExecutor` of Python callables) because the work is *waiting on external processes*: no pickling, no `__main__` guard, no parent re-import — which sidesteps the entire Windows-`spawn` failure class the debate's Critic raised. The canonical reproducibility run uses `--jobs 1` (quiet CPU → clean `wallclock_per_step`); the researcher opts into `--jobs 8` for the multi-hour Phase 6/6b sweeps. **Because outputs are order-independent, episode result bytes are identical at any `--jobs` value.** Rejected: parallel-by-default (perturbs the `wallclock_per_step` freebie on the canonical run, against the project's reproducibility-first priority); in-process loop (reintroduces irsim/matplotlib global-state bleed — the repo's recurring `id_iter`/`WrapTo2Pi` failure class).
- **Overwrite by default; `--resume` to skip existing `<seed>.json`** — For a reproducibility study, "the whole set came from this master seed in this run" is the clean guarantee. `--resume` is the explicit opt-in for cheap re-runs of missing seeds. Rejected: resume-by-default (risks silently mixing results across code versions / master seeds).
- **Failure policy: continue + report; batch exits non-zero iff any seed's subprocess exited non-zero** — A single bad seed must not waste the batch. **Crucial semantic:** a subprocess exit of 0 includes in-sim crashes, timeouts, and planner failures (those are recorded *in the metrics JSON*, not the exit code). The batch only flags **runner failures** (non-zero subprocess exit). "succeeded" in the summary means "the episode ran to completion," NOT "the robot reached the goal." Rejected: abort-on-first-failure (loses partial progress on a run-once job).
- **One algorithm + one world per invocation** — `--algorithm <name> --world <yaml> [--master-seed N] [--num-seeds 50] [--jobs N] [--results-dir DIR] [--resume] [--traffic|--no-traffic]`. Matches the single-episode runner's shape; run once per algorithm. Rejected: loop-all-registered (only `a_star_once` exists today; premature).
- **Master seed: a module-level `DEFAULT_MASTER_SEED` constant, `--master-seed` override** — "One master seed per experiment" with a pinned canonical default so an argument-free invocation is reproducible. Default value is an arbitrary fixed int (e.g. `20260605`); its specific value carries no meaning beyond "the canonical stream."
- **Deterministic `_manifest.json` provenance receipt (console carries timing)** — Written to `results/<world_stem>/<algorithm>/_manifest.json`. Records `master_seed`, `num_seeds`, `algorithm`, `world`, `world_stem`, `traffic`, best-effort `git_sha` (or `null`), `derived_seeds` (length `num_seeds`), and `episodes` (length `num_seeds`, each `{seed, exit_code, status}`, **ordered by derivation index** so order is `--jobs`-independent). It **excludes wall-clock and timestamps** so it is byte-identical across two same-master-seed runs at the same commit; human-facing elapsed time goes to the console summary only. Rejected: embedding a timestamp/elapsed field (breaks the determinism the receipt is meant to certify).
- **Manifest co-located but glob-distinguishable** — `_manifest.json` lives in the algorithm dir beside the `<seed>.json` episode files. Because derived seeds are large all-digit integers, Phase 5's `plot.py` must select episode files by numeric stem (e.g. glob `[0-9]*.json`) or skip `_`-prefixed files — a forward-compat breadcrumb, not Phase 3 work. Rejected: a naive `*.json` glob in Phase 5 (would ingest `_manifest.json` as a fake episode).
- **`run_experiment` validates `--algorithm` and `--world` up front, then launches** — Imports `ALGORITHMS` from `runners.run_episode` to reject an unknown algorithm before spawning 50 subprocesses; checks `Path(world).exists()` (existence only, not a full parse — a malformed-but-existing YAML still reaches the child, which is what TC27 exploits). `--num-seeds >= 1`, `--jobs >= 1` validated.
- **New `--check` TCs are fast and fully deterministic; the slow full run is manual** — Measured `--check` is **~9–10 min** today (545.8 s), already far over the stale "120 s" claim; a full 50-seed×2 reproducibility run is ~50–100 min and cannot live in `--check`. So TC25/TC26/TC27 are budget-light (derivation math is instant; TC26/TC27 use a **fast-failing world** so episodes terminate in seconds without a driving loop). The full-scale reproducibility + parallel-determinism check is a documented manual step (T-Verify) that uses `--traffic` (A* crashes fast → short episodes) and `--jobs 8` to stay tractable.
- **Ships as one PR** — Module + 3 TCs + docs are tightly coupled (TCs depend on the module; docs depend on the final CLI). One branch, one PR.

## Acceptance Criteria

- [ ] **AC1 (happy path):** `python -m runners.run_experiment --algorithm a_star_once --world arena/arena_v1.yaml --num-seeds 3 --no-traffic --results-dir <tmp>` exits 0, prints a per-seed progress line and an end summary, and writes — for each of the 3 derived seeds — `<tmp>/arena_v1/a_star_once/<seed>.json` and `<tmp>/arena_v1/a_star_once/<seed>.trace.jsonl`, plus `<tmp>/arena_v1/a_star_once/_manifest.json`.
- [ ] **AC2 (seed derivation):** `derive_episode_seeds(M, 50)` returns 50 distinct **non-negative** Python ints (the 64-bit pack is in `[0, 2^64)`; zero is theoretically possible and not an error); `derive_episode_seeds(M, 3) == derive_episode_seeds(M, 50)[:3]` (prefix property); calling it twice with the same `M` returns the identical tuple; a contrived collision raises a clear error rather than silently dropping a seed.
- [ ] **AC3 (batch reproducibility, `--jobs 1`):** Two `run_experiment` invocations with the **same** `--master-seed`, `--num-seeds`, `--world arena/arena_no_path.yaml`, `--no-traffic`, `--jobs 1`, into two different `--results-dir`, produce, for every derived seed: byte-identical `<seed>.json` (the planner-failure path hardcodes `wallclock_per_step=0.0`, so the metrics JSON is fully deterministic here), and byte-identical `_manifest.json`. (On a *success* world, the metrics JSON differs only in `wallclock_per_step`; the trace JSONL stays byte-identical — see AC4.)
- [ ] **AC4 (parallel determinism — manual/T-Verify):** A full default run (`--num-seeds 50`, `--traffic`) at `--jobs 1` vs `--jobs 8` produces, per seed: byte-identical `trace.jsonl`, and metrics JSON equal in **every field except `wallclock_per_step`**; `_manifest.json` deterministic fields (`derived_seeds`, `episodes[].seed`, ordering) identical. Demonstrates the parallel speedup does not alter result bytes.
- [ ] **AC5 (failure accounting):** When ≥1 seed's `run_episode` subprocess exits non-zero, the batch runs every remaining seed, lists each failed seed (seed value, exit code, stderr tail) in the summary, records `status == "runner_error"` for it in the manifest, and the `run_experiment` process itself exits non-zero. When all seeds succeed, it exits 0.
- [ ] **AC6 (re-run policy):** With no flag, a second run **overwrites** existing `<seed>.json`/`.trace.jsonl`. With `--resume`, seeds whose `<seed>.json` already exists are **skipped** (counted as "skipped" in the summary and manifest) and only missing seeds run.
- [ ] **AC7 (manifest schema):** `_manifest.json` parses to an object with exactly `master_seed:int`, `num_seeds:int`, `algorithm:str`, `world:str`, `world_stem:str`, `traffic:bool`, `git_sha:str|null`, `derived_seeds:[int]` (length `num_seeds`), `episodes:[{seed:int, exit_code:int, status:str}]` (length `num_seeds`, ordered by derivation index). No timestamp/elapsed key.
- [ ] **AC8 (`--check`):** `python arena/arena.py arena/arena_v1.yaml --check` reports **28 PASS** lines (the existing 25 + TC25 + TC26 + TC27) and exits 0. The three new TCs add only a few seconds (TC25 instant; TC26/TC27 use a fast-failing world). The suite total is ~9–10 min; **no "under 120 s" claim is asserted or reintroduced.**
- [ ] **AC9 (docs):** `CLAUDE.md` gains a "## The batch experiment runner (Phase 3)" section (CLI, seed-derivation rule, `--jobs`/`--resume`/failure semantics, manifest, results layout, TC25–TC27 one-liners). The stale "`--check` … under 120 s" / "25 PASS" wording in the existing arena + runner sections is corrected to the measured runtime and the new 28-PASS count. `Mission.md` optionally gains a one-line "batch runner landed" note.
- [ ] **AC10 (no collateral changes):** `runners/run_episode.py` and `arena/arena.py`'s `Arena` class gain **no behavioral change** (run_experiment composes over them; arena.py changes are test-only TC additions). Existing TC1–TC24 still PASS.

## Data Model

```python
# runners/run_experiment.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

DEFAULT_MASTER_SEED = 20260605   # canonical experiment stream; value is arbitrary
DEFAULT_NUM_SEEDS = 50           # Mission.md: 50 seeds per algorithm
MANIFEST_NAME = "_manifest.json" # underscore prefix; Phase 5 globs episodes by numeric stem


def derive_episode_seeds(master_seed: int, num_seeds: int) -> tuple[int, ...]:
    """Master seed -> N decorrelated 64-bit episode seeds via SeedSequence.spawn.

    Each child's first two uint32 state words are packed into a 64-bit int.
    Deterministic, prefix-stable (spawn(3) == spawn(50)[:3]), uniqueness-asserted.
    """
    if num_seeds < 1:
        raise ValueError(f"num_seeds must be >= 1, got {num_seeds}")
    children = np.random.SeedSequence(master_seed).spawn(num_seeds)
    seeds: list[int] = []
    for child in children:
        w = child.generate_state(2, dtype=np.uint32)
        seeds.append(int(w[0]) | (int(w[1]) << 32))
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(
            f"derived seed collision at master={master_seed}, num_seeds={num_seeds}"
        )
    return tuple(seeds)


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    exit_code: int            # subprocess return code; 0 == ran to completion
    status: str               # "ok" | "runner_error" | "skipped"
    stderr_tail: str          # last few lines of child stderr (empty when ok)


@dataclass(frozen=True)
class BatchSummary:
    n_total: int
    n_ok: int
    n_failed: int
    n_skipped: int
    exit_code: int            # 0 iff n_failed == 0


def summarize(results: tuple[EpisodeResult, ...]) -> BatchSummary:
    """Pure tally over per-seed results. No I/O — unit-testable (TC27)."""
    ...
```

```jsonc
// results/<world_stem>/<algorithm>/_manifest.json — deterministic provenance receipt
{
  "master_seed": 20260605,
  "num_seeds": 50,
  "algorithm": "a_star_once",
  "world": "arena/arena_v1.yaml",
  "world_stem": "arena_v1",
  "traffic": true,
  "git_sha": "2a7246d…",          // best-effort; null if not a git checkout
  "derived_seeds": [/* 50 ints */],
  "episodes": [                    // ordered by derivation index (jobs-independent)
    {"seed": 13452890127, "exit_code": 0, "status": "ok"}
    // …
  ]
}
```

## API Contracts

`run_experiment` CLI:

```
python -m runners.run_experiment
    --algorithm <name>        required; validated against runners.run_episode.ALGORITHMS
    --world <yaml>            required; Path(world).exists() checked up front
    [--master-seed <int>]    default DEFAULT_MASTER_SEED
    [--num-seeds <int>]      default 50; >= 1
    [--jobs <int>]           default 1 (sequential); N>1 => ThreadPoolExecutor(max_workers=N)
    [--results-dir <dir>]    default "results"; forwarded to each run_episode
    [--resume]               skip seeds whose <seed>.json already exists (default: overwrite)
    [--traffic|--no-traffic] forwarded to run_episode; default --traffic

Per-seed child invocation (exactly the shipping single-episode entry point):
    sys.executable -m runners.run_episode
        --algorithm <name> --seed <derived_int> --world <yaml>
        --results-dir <dir> [--no-traffic]
  launched via subprocess.run(..., capture_output=True, text=True, cwd=<repo_root>)

Execution:
    out_dir = <results-dir>/<world_stem>/<algorithm>   # pre-created ONCE before dispatch
    seeds   = derive_episode_seeds(master_seed, num_seeds)
    jobs==1 : iterate seeds in order
    jobs>1  : ThreadPoolExecutor(max_workers=jobs); submit all; collect via as_completed
    results assembled in DERIVATION-INDEX order for the manifest (not completion order)

Exit codes:
    0  — every non-skipped seed's subprocess exited 0
    1  — >= 1 seed's subprocess exited non-zero (continue-and-report)
    2  — argparse error / up-front validation failure (unknown algorithm, missing world)
```

## Error Handling

- **Unknown `--algorithm`** — argparse `choices=list(ALGORITHMS)` rejects before any subprocess launches (exit 2).
- **`--world` path missing** — up-front `Path(world).exists()` check fails fast (exit 2). A malformed-but-existing YAML passes this check and fails *inside the child* (the mechanism TC27 uses).
- **A seed's child subprocess exits non-zero** — captured `(exit_code, stderr_tail)`; the seed is marked `status="runner_error"`; the batch continues; the summary lists it; `run_experiment` returns 1.
- **`--resume` with a half-written prior `<seed>.json`** — `--resume` skips on *file existence*, so a truncated file from a crashed prior run would be skipped. Documented trade-off; overwrite (the default) is the safe choice and avoids it.
- **`git rev-parse HEAD` fails** (not a repo / git absent) — `git_sha` recorded as `null`; the batch proceeds. Best-effort provenance only.
- **`--jobs` greater than CPU count** — allowed; print a one-line advisory (`jobs=N exceeds cpu_count=M`) but do not clamp. Oversubscription only perturbs `wallclock_per_step` (a Mission.md "freebie"); result bytes are unaffected.
- **`wallclock_per_step` under `--jobs > 1`** — CPU contention scatters this metric (already flagged non-deterministic in CLAUDE.md). All other metrics fields and the entire trace JSONL remain byte-identical regardless of `--jobs`. Produce headline wallclock numbers with `--jobs 1` if that telemetry matters.
- **Concurrent `mkdir` race (Windows)** — avoided: `run_experiment` pre-creates `out_dir` once before dispatching any worker, so children never race on the shared parent (each child's own `mkdir(exist_ok=True)` then no-ops).

## Testing Strategy

**Levels:** Unit (`derive_episode_seeds`, `summarize`), Integration (`run_experiment` end-to-end via subprocess on fast worlds), Regression (TC1–TC24 unchanged). All as `TCi` functions in `arena/arena.py --check`; no pytest. New TCs use **fast-failing / fast-construction** worlds so they add only seconds to the ~9–10 min suite.

| ID   | Test Case | Type | Expected Behavior |
|------|-----------|------|-------------------|
| TC25 | Seed-derivation determinism, uniqueness, prefix | Unit | `derive_episode_seeds(7, 50)` has 50 distinct non-negative ints; equals itself on a second call; `derive_episode_seeds(7, 3) == derive_episode_seeds(7, 50)[:3]`; a different master seed yields a different tuple. Pure computation — no subprocess. |
| TC26 | Batch orchestration + parallel-ordering determinism | Integration | Subprocess-invoke `run_experiment --algorithm a_star_once --world arena/arena_no_path.yaml --num-seeds 3 --no-traffic` **three** times into three tempdirs: run A `--jobs 1`, run B `--jobs 1`, run C `--jobs 3`. The start `(2,2)` is boxed in so A* raises immediately → each episode terminates in seconds with `planner_error` set and `wallclock_per_step=0.0`. Assert: all exit 0; for each derived seed `<seed>.json` is byte-identical across A and B; the manifests of A and B are byte-identical; the manifest of C (`--jobs 3`) has identical `derived_seeds` and identical `episodes` ordering (by derivation index) to A — proving completion-order under parallelism does NOT reorder the manifest. Manifest comparisons load JSON and drop `git_sha` before comparing, so the test is robust to a dirty tree / absent git. No `<seed>.trace.jsonl` exists (planner-failure contract). |
| TC27 | Failure accounting + non-zero batch exit | Integration | Write a malformed `_tmp_bad.yaml` (exists, but irsim rejects it) in a tempdir; subprocess-invoke `run_experiment --algorithm a_star_once --world <_tmp_bad.yaml> --num-seeds 2 --no-traffic --jobs 1`. Assert: `run_experiment` exits non-zero; its stdout summary names 2 failed seeds; `_manifest.json` `episodes` both have `status=="runner_error"` with non-zero `exit_code`. Confirms continue-and-report + batch exit propagation. |

**Test data:** TC26 uses the existing `arena/arena_no_path.yaml` (the robot **start** `(2,2)` is walled inside a 1.5 m box of four rectangles → A* can't escape → fast failure, no driving loop; the goal `(48,48)` is open). TC27 writes a throwaway malformed YAML to a tempdir. No new fixture files.

**Run commands:**

```powershell
.venv\Scripts\Activate.ps1
python arena/arena.py arena/arena_v1.yaml --check        # 28 PASS (25 existing + TC25-TC27); ~9-10 min total

# Phase 3 deliverable — full reproducible batch for one algorithm:
python -m runners.run_experiment --algorithm a_star_once --world arena/arena_v1.yaml
        # default: master-seed 20260605, 50 seeds, traffic ON, jobs 1 (sequential)
        # writes results/arena_v1/a_star_once/<seed>.{json,trace.jsonl} x50 + _manifest.json

# Faster sweep run (opt-in parallelism; identical result bytes):
python -m runners.run_experiment --algorithm a_star_once --world arena/arena_v1.yaml --jobs 8

# Resume only the missing seeds:
python -m runners.run_experiment --algorithm a_star_once --world arena/arena_v1.yaml --resume
```

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T0 | **Create feature branch `phase3-reproducibility` from updated `main`.** Phase 2 was merged to `main` (origin/main @ `488d84e`, PR #1), so `main` now contains `run_episode.py` / `arena/dynamic.py` / traffic. Fetch and branch from `origin/main`; the PR targets `main`. | — | low | (git) | `git fetch origin && git checkout -b phase3-reproducibility origin/main`. Project rule: never commit to main directly. |
| T1 | **Implement `runners/run_experiment.py`** per the Data Model + API Contracts. Includes: `DEFAULT_MASTER_SEED`/`DEFAULT_NUM_SEEDS`/`MANIFEST_NAME` constants; `derive_episode_seeds(master, n)` (SeedSequence.spawn → 64-bit pack → uniqueness assert); frozen `EpisodeResult`/`BatchSummary`; pure `summarize(results)`; argparse CLI (`--algorithm` validated against imported `runners.run_episode.ALGORITHMS`, `--world` existence-checked, `--master-seed`/`--num-seeds`/`--jobs`/`--results-dir`/`--resume`, mutually-exclusive `--traffic`/`--no-traffic` default ON); pre-create `out_dir` once; per-seed `subprocess.run([sys.executable,'-m','runners.run_episode',…], capture_output=True, text=True, cwd=repo_root)`; sequential when `jobs==1`, `ThreadPoolExecutor(max_workers=jobs)` + `as_completed` when `jobs>1`; `--resume` skip-on-`<seed>.json`-exists; continue-and-report; assemble results in derivation-index order; best-effort `git_sha` via `git rev-parse HEAD`; write `_manifest.json` (no timestamp/elapsed); console progress + end summary (counts + elapsed); `main(argv)->int` returning 0/1/2; `if __name__=='__main__': raise SystemExit(main())`. Match `runners/run_episode.py` style (frozen dataclasses, type hints, `from __future__ import annotations`, repo-root `sys.path` bootstrap). Satisfies AC1, AC2, AC4, AC5, AC6, AC7, AC10. | T0 | high | `runners/run_experiment.py` (new) | **Determinism rule:** the manifest's `episodes` and `derived_seeds` MUST be in derivation-index order, NOT subprocess-completion order — else `--jobs` perturbs the manifest bytes (breaks AC3/TC26). **Threads, not processes:** use `ThreadPoolExecutor` over `subprocess.run`; do NOT use `ProcessPoolExecutor`/`multiprocessing` (Windows `spawn` re-imports the parent and the runner mutates `sys.path` at import — a known footgun). **"succeeded" == subprocess exit 0**, which includes in-sim crashes/timeouts/planner-failures; only non-zero exits are runner failures. Pack seeds as `int(w0) | (int(w1)<<32)` exactly (64-bit) — 32-bit risks a silent same-filename collision. |
| T2 | **Add TC25, TC26, TC27 to `arena/arena.py`** and register them in `_run_checks` (after TC24); **also update arena.py's own `--check` argparse help / docstring TC-count wording** (e.g. the `"Run TC1-TC24 … (25 cases)"` help string) to TC1–TC27 / 28 cases. TC25: unit-assert `derive_episode_seeds` determinism/uniqueness/prefix (import the helper from `runners.run_experiment`). TC26: subprocess-invoke `run_experiment` on `arena/arena_no_path.yaml` (`--num-seeds 3 --no-traffic`) **three** times — A `--jobs 1`, B `--jobs 1`, C `--jobs 3` — into three tempdirs; assert all exit 0, byte-identical per-seed `<seed>.json` (A vs B), byte-identical manifests A vs B and identical `derived_seeds` + `episodes` ordering A vs C (parallel doesn't reorder), no trace files. **Manifest comparisons drop `git_sha` before comparing** (robust to dirty tree / no git). TC27: write a malformed YAML to a tempdir, invoke `run_experiment` on it; assert non-zero exit, summary names 2 failures, manifest `episodes` both `status=="runner_error"` with non-empty `stderr_tail`. Mirror the existing subprocess-TC scaffolding (TC15/TC22). Satisfies AC2, AC3, AC5, AC7, AC8. | T1 | med | `arena/arena.py` | Use `filecmp.cmp(..., shallow=False)` or byte-read for the per-seed-JSON identity checks (mirrors TC15); load+drop-`git_sha` for manifest compares. TC26/TC27 each spawn nested subprocesses (`run_experiment` → `run_episode`); on the boxed-in-start / malformed worlds each child terminates in seconds, but budget for ~15–40 s per TC. **Verify the new PASS-line count empirically** — the suite currently emits **25** PASS lines for TC1–TC24 (TC2b adds an extra line), so the target is **28** lines; if the baseline differs, update AC8 and the docs to match the real count rather than assuming. |
| T3 | **Docs: add the Phase 3 section to `CLAUDE.md` and correct the stale `--check` claim.** New "## The batch experiment runner (Phase 3)" section: CLI surface, `derive_episode_seeds` rule (master → spawn(50) → 64-bit ints), `--jobs`/`--resume`/failure semantics, the `_manifest.json` receipt, the `results/<world_stem>/<algorithm>/` layout (and the Phase-5 `[0-9]*.json` glob breadcrumb), and one-liners for TC25–TC27. **Also fix the existing arena + runner sections of `CLAUDE.md`**: change the "`25 PASS … (TC1-TC24 …)`" / "`24 PASS`" / "`under 120 s`" wording to the measured `--check` runtime (~9–10 min, dominated by the full-episode subprocess TCs — do NOT hardcode an exact second count) and the new **28 PASS** count. Optionally add a one-line "Phase 3 batch runner landed" note to `Mission.md`. Satisfies AC9. | T1, T2 | low | `CLAUDE.md`, `Mission.md` | Blocked by T2 because it documents the **28 PASS** count and the arena.py help-string change T2 makes. Keep the new section ~20–30 lines, matching the existing Phase 1/2 sections. The "under 120 s" correction appears in more than one place — grep `120` and `PASS` in CLAUDE.md and fix each occurrence consistently. |
| T4 | **Manual verification** (gate for "Phase 3 done"). Activate venv. Run `python arena/arena.py arena/arena_v1.yaml --check` → expect **28 PASS**, exit 0. Run `python -m runners.run_experiment --algorithm a_star_once --world arena/arena_v1.yaml --num-seeds 5 --no-traffic --results-dir _tmp_p3a --jobs 1`, then again into `_tmp_p3b`, and confirm per-seed `trace.jsonl` byte-identical + metrics equal-except-`wallclock_per_step` + manifests' deterministic fields equal. Then the **parallel-determinism check** (AC4): one full default run (`--traffic`, jobs 1) into `_tmp_seq` and one with `--jobs 8` into `_tmp_par`; confirm per-seed trace byte-identical and metrics equal-except-`wallclock_per_step` (use `--traffic` so A* crashes fast → tractable wall-clock; `--jobs 8` keeps it ~6–10 min). Delete all `_tmp_*`. Record the AC4 outcome (held / fell back) in the PR description. Satisfies AC3 (full-scale), AC4. | T2, T3 | low | (observation only) | Provide the equal-except-`wallclock_per_step` comparison as a short inline python snippet (load both JSONs, pop `wallclock_per_step`, assert equal). If any per-seed trace diverges across the two runs, the determinism break is in `run_episode.py`/Arena, NOT `run_experiment.py` — `run_experiment` only chooses seeds and schedules subprocesses. `_tmp_*` is gitignored per project convention. |

## Notes for Implementer

- **The whole point of Phase 3** is "same 50 traffic streams for every algorithm." The single master seed + `SeedSequence.spawn(50)` *is* that guarantee — do not let any per-run randomness (timestamps, PID, wall-clock) leak into the seed derivation or the manifest's deterministic fields.
- **Prefix property** (`spawn(3) == spawn(50)[:3]`) holds only because each derivation starts from a **fresh** `SeedSequence(master)` and calls `spawn(num_seeds)` exactly once. Do not cache or reuse a spawned `SeedSequence` across calls.
- **64-bit pack, then assert unique.** `int(w0) | (int(w1) << 32)`. The uniqueness assertion is belt-and-suspenders for an event that should never occur at 64-bit width; if it ever fires, it fires *reproducibly* (same master seed) rather than silently producing 49 files.
- **Manifest ordering is determinism-critical.** Build `episodes` from the derivation-index list, not from `as_completed` order. Under `--jobs > 1`, completion order is nondeterministic; derivation order is not.
- **Threads over processes for `--jobs`.** `ThreadPoolExecutor` blocks on `subprocess.run`; the GIL is released during the wait, there is no pickling, and the Windows `spawn` re-import footgun (the runner and arena both mutate `sys.path` at import time) never triggers. This was the explicit resolution of the execution-model `/debate`.
- **`as_completed` loses the index — carry it.** When `--jobs > 1`, submit with a `{future: derivation_index}` (or `{future: seed}`) map and, as each future completes, store its `EpisodeResult` into a pre-sized list at its derivation index. Build the manifest from that index-ordered list, never from completion order. This is the exact spot the ordering bug TC26's `--jobs 3` arm guards against.
- **Resolve `--world` to an absolute path once, up front**, and forward the absolute path to every child `--world`. `run_experiment`'s own `Path(world).exists()` check and the child subprocesses (launched with `cwd=repo_root`) must agree on the path even if `run_experiment` was invoked from a subdirectory. The manifest records the absolute world path; `world_stem = Path(world).stem` is unaffected.
- **`stderr_tail` is bounded.** Capture only the last ~20 lines (or ~2 KB) of a failed child's stderr for the summary/manifest, not the whole buffer — a child under traffic can emit a lot. TC27 asserts the tail is non-empty on a real failure so the failure-reporting UX can't silently regress to empty strings.
- **A *seed* cannot cause a non-zero subprocess exit.** `run_episode` returns 0 for in-sim crashes, timeouts, and planner failures alike. The only non-zero paths are argparse/`__init__` config errors — which, with `--algorithm` and `--world`-existence pre-validated by `run_experiment`, reduce to "a world that exists but irsim rejects" (TC27's mechanism) or an irsim/Arena init fault. So "continue past a failed seed" really means "continue past a runner/config fault that would otherwise abort the batch"; phrase the summary accordingly.
- **PR hygiene:** per the user-global rules, the commit messages and PR body must read as human-authored — no `Co-Authored-By`, no "Generated with Claude Code" footer, no AI tells. Match the repo's existing `type(scope): subject` commit voice (see `git log`); PR body is a short plain paragraph.
- **`run_episode` exit 0 ≠ goal reached.** In-sim crashes, timeouts, and planner failures all return 0 (their disposition is in the metrics JSON). The batch's failure tally counts only non-zero subprocess exits. Make the summary wording reflect this ("ran to completion", not "succeeded at the task").
- **`arena_no_path.yaml` is the fast-failure fixture for TC26** — the robot **start** `(2,2)` is boxed in by four 1.5 m rectangle walls (the goal is open), so A* raises immediately and the episode never enters the driving loop; `wallclock_per_step` is hardcoded `0.0` in the planner-failure branch (`run_episode.py:281`), making the metrics JSON fully byte-deterministic. That is why TC26 can assert byte-identical `<seed>.json` (a *success* world could not, because of `wallclock_per_step`).
- **The `--check` budget is already blown vs its doc claim.** Measured 545.8 s for 25 PASS today; the "under 120 s" wording is stale. Phase 3 corrects the doc rather than trying to make the suite fit 120 s. Keep the new TCs fast (no full driving episodes) so the suite doesn't grow meaningfully.
- **Phase 5 breadcrumb:** episode files have all-digit stems; the manifest is `_manifest.json`. When Phase 5's `plot.py` is written, it must glob `[0-9]*.json` (or skip `_`-prefixed files) so the manifest isn't mistaken for an episode. This is a comment/docs note now, not code.
- **Rollback:** delete `runners/run_experiment.py`; revert the TC25–TC27 additions and `_run_checks` registration in `arena/arena.py`; revert the `CLAUDE.md`/`Mission.md` doc edits; delete the `phase3-reproducibility` branch. Repo returns to its Phase 2 state.
- **What this plan deliberately does NOT do:** compute per-algorithm aggregates or failure rates (Phase 4), draw the scatter plot (Phase 5), add any planner or the K-sweep (Phase 6/6b), or change `Arena`'s seed API. If the executor finds themselves writing aggregation math or editing `Arena.__init__`, stop — that's a different plan.
