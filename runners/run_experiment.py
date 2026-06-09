"""Batch experiment runner — runs one algorithm against the canonical 50 seeds.

Phase 3 (Reproducibility). Mission.md requires every algorithm to face the SAME
50 traffic streams so the cross-algorithm scatter plot is meaningful. This module
derives 50 per-episode seeds from a single master seed via `SeedSequence.spawn`,
then launches the already-deterministic single-episode runner
(`runners.run_episode`) once per seed — one fresh-irsim subprocess each, so per-file
byte-determinism (proven by TC15/TC20/TC24) and the spawn-based traffic/motion
substreams carry over unchanged.

CLI:
    python -m runners.run_experiment \
        --algorithm <name>      # required; e.g. "a_star_once"
        --world <yaml_path>     # required; e.g. arena/arena_v1.yaml
        [--replan-k <int>]      # required for the _replan family, forbidden otherwise
        [--master-seed <int>]   # default DEFAULT_MASTER_SEED
        [--num-seeds <int>]     # default 50; >= 1
        [--jobs <int>]          # default 1 (sequential); N>1 => bounded concurrency
        [--results-dir <dir>]   # default "results"; forwarded to each episode
        [--resume]              # skip seeds whose <seed>.json already exists
        [--traffic|--no-traffic]# Phase 2 crossing traffic, default ON

Outputs (per seed, written by the child run_episode):
    <results-dir>/<world_stem>/<label>/<seed>.json
    <results-dir>/<world_stem>/<label>/<seed>.trace.jsonl   (only on planning success)
Plus a provenance receipt written by this module:
    <results-dir>/<world_stem>/<label>/_manifest.json
where <label> = algorithm_label(<algorithm>, <replan-k>) (e.g. "a_star_once",
"a_star_replan_k5"), so replan cadences do not collide.

Execution:
    --jobs 1 (default) runs seeds sequentially. --jobs N>1 runs up to N child
    subprocesses concurrently via a ThreadPoolExecutor (threads waiting on
    subprocess.run — no multiprocessing, so the Windows spawn/pickle path never
    enters). Each seed is isolated, so the per-seed trace JSONL and the manifest
    are byte-identical at any --jobs value; the metrics JSON matches too EXCEPT
    `wallclock_per_step` (a Mission.md "freebie"), a perf_counter mean that
    contention perturbs. Produce headline wallclock numbers with --jobs 1.

    Caveat at extreme oversubscription: EPISODE_TIMEOUT_S is a per-child wallclock
    wall, so a healthy-but-starved child could in principle exceed it under very
    high --jobs and flip ok -> runner_error. That can't happen with today's
    fail-fast `_once` planners (the in-sim 120 s cap ends them first); revisit the
    wall for future real-driving algorithms.

Exit codes:
    0 — every non-skipped seed's subprocess exited 0 (ran to completion)
    1 — >= 1 seed's subprocess exited non-zero (continue-and-report)
    2 — argparse error / up-front validation failure (unknown algorithm, missing world,
        bad --replan-k for the chosen family)

Note: a child exit of 0 includes in-sim crashes, timeouts, and planner failures
(those are recorded inside the metrics JSON, not the exit code). "succeeded" below
means "ran to completion", NOT "the robot reached the goal".
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Make repo root importable so `runners.run_episode` resolves when this module is
# invoked as `python -m runners.run_experiment` from any cwd. Mirrors
# runners/run_episode.py:44-48.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planners import algorithm_label, build_controller  # noqa: E402
from runners._layout import episode_out_dir  # noqa: E402
from runners.run_episode import ALGORITHMS  # noqa: E402


DEFAULT_MASTER_SEED = 20260605          # canonical experiment stream; value is arbitrary
DEFAULT_NUM_SEEDS = 50                   # Mission.md: 50 seeds per algorithm
DEFAULT_RESULTS_DIR = "results"
MANIFEST_NAME = "_manifest.json"         # underscore prefix; Phase 5 globs episodes by numeric stem
STDERR_TAIL_LINES = 20                   # last N lines of a failed child's stderr, for the console
EPISODE_TIMEOUT_S = 600.0                # hard wall on one child; the in-sim cap is 120 s sim-time,
                                         # so ~10 min wallclock means the child is wedged, not slow
TIMEOUT_EXIT_CODE = 124                  # synthetic exit code recorded for a timed-out child (mirrors GNU timeout)
SPAWN_ERROR_EXIT_CODE = 125              # synthetic exit code for a child that never started (spawn OSError)
GIT_SHA_TIMEOUT_S = 10.0                  # hard wall on the manifest git-sha probe; a wedged git
                                          # (held index.lock, slow network FS) must not hang the batch


def derive_episode_seeds(master_seed: int, num_seeds: int) -> tuple[int, ...]:
    """Master seed -> N decorrelated 64-bit episode seeds via SeedSequence.spawn.

    Each child SeedSequence's first two uint32 state words are packed into a
    64-bit int. Deterministic and prefix-stable: because each child is seeded
    only by its spawn-key ``(i,)``, ``derive_episode_seeds(M, 3)`` equals
    ``derive_episode_seeds(M, 50)[:3]``. 64-bit width keeps the birthday-collision
    probability over 50 draws negligible; the uniqueness check makes even that
    impossible case loud and reproducible rather than a silent same-filename clash.
    """
    if num_seeds < 1:
        raise ValueError(f"num_seeds must be >= 1, got {num_seeds}")
    children = np.random.SeedSequence(master_seed).spawn(num_seeds)
    seeds: list[int] = []
    for child in children:
        words = child.generate_state(2, dtype=np.uint32)
        seeds.append(int(words[0]) | (int(words[1]) << 32))
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(
            f"derived seed collision at master={master_seed}, num_seeds={num_seeds} "
            "(astronomically unlikely at 64-bit width — investigate before retrying)"
        )
    return tuple(seeds)


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of one seed's child subprocess."""

    seed: int
    exit_code: int          # subprocess return code; 0 == ran to completion
    status: str             # "ok" | "runner_error" | "skipped"
    stderr_tail: str        # last lines of child stderr on failure; "" otherwise (console only)


@dataclass(frozen=True)
class BatchSummary:
    """Aggregate tally over all per-seed results."""

    n_total: int
    n_ok: int
    n_failed: int
    n_skipped: int
    exit_code: int          # 0 iff n_failed == 0


def summarize(results: tuple[EpisodeResult, ...]) -> BatchSummary:
    """Pure tally over per-seed results. No I/O — unit-testable (TC27/TC25 adjacent)."""
    n_ok = sum(1 for r in results if r.status == "ok")
    n_failed = sum(1 for r in results if r.status == "runner_error")
    n_skipped = sum(1 for r in results if r.status == "skipped")
    return BatchSummary(
        n_total=len(results),
        n_ok=n_ok,
        n_failed=n_failed,
        n_skipped=n_skipped,
        exit_code=0 if n_failed == 0 else 1,
    )


def _git_sha(repo_root: Path) -> str | None:
    """Best-effort HEAD sha for the manifest; None if not a git checkout / git absent."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=GIT_SHA_TIMEOUT_S,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    sha = r.stdout.strip()
    return sha or None


def _stderr_tail(text: str, n_lines: int = STDERR_TAIL_LINES) -> str:
    """Last ``n_lines`` lines of ``text`` (bounded console output; not stored in the manifest)."""
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:])


def _run_one_episode(
    *,
    seed: int,
    algorithm: str,
    replan_k: int | None,
    world_abs: str,
    results_dir: str,
    traffic: bool,
    repo_root: Path,
) -> EpisodeResult:
    """Launch one `python -m runners.run_episode` subprocess for a single seed."""
    cmd = [
        sys.executable,
        "-m",
        "runners.run_episode",
        "--algorithm",
        algorithm,
        "--seed",
        str(seed),
        "--world",
        world_abs,
        "--results-dir",
        results_dir,
    ]
    # Forward the cadence only when set; a non-replan family must NOT see the flag
    # (run_episode rejects --replan-k for those, so passing None would be wrong).
    if replan_k is not None:
        cmd.extend(["--replan-k", str(replan_k)])
    cmd.append("--traffic" if traffic else "--no-traffic")
    # capture_output buffers the child's full stdout/stderr though only the last
    # STDERR_TAIL_LINES are surfaced; that is intentional — we need the tail on
    # failure, and a child's output is small (a few lines), so streaming to keep
    # only the tail would add machinery for no real memory win.
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=EPISODE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        # A child that blows the wallclock wall is wedged (not slow — the in-sim 120 s
        # cap would have ended it). Record it as a runner failure so the batch continues.
        return EpisodeResult(
            seed=seed,
            exit_code=TIMEOUT_EXIT_CODE,
            status="runner_error",
            stderr_tail=f"timed out after {EPISODE_TIMEOUT_S:.0f}s (child killed)",
        )
    except OSError as exc:
        # The child never started: a spawn-level failure such as descriptor/handle
        # exhaustion under high --jobs or a missing interpreter. Record it as a runner
        # failure (not a crash) so the batch continues and the accounting for the seeds
        # that already finished still reaches the manifest — the documented contract.
        return EpisodeResult(
            seed=seed,
            exit_code=SPAWN_ERROR_EXIT_CODE,
            status="runner_error",
            stderr_tail=f"failed to spawn child: {exc}",
        )
    if proc.returncode == 0:
        return EpisodeResult(seed=seed, exit_code=0, status="ok", stderr_tail="")
    return EpisodeResult(
        seed=seed,
        exit_code=proc.returncode,
        status="runner_error",
        stderr_tail=_stderr_tail(proc.stderr or proc.stdout or ""),
    )


@dataclass(frozen=True)
class RunnerArgs:
    """Parsed CLI arguments — frozen so accidental mutation is impossible."""

    algorithm: str
    world: str
    replan_k: int | None
    master_seed: int
    num_seeds: int
    jobs: int
    results_dir: str
    resume: bool
    traffic: bool


def _parse_args(argv: list[str] | None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        prog="runners.run_experiment",
        description="Run one algorithm against the canonical 50 seeds (Phase 3 batch runner).",
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=list(ALGORITHMS),
        help="Registered planner name (e.g. 'a_star_once').",
    )
    parser.add_argument(
        "--world",
        required=True,
        help="Path to the world YAML (e.g. arena/arena_v1.yaml).",
    )
    parser.add_argument(
        "--replan-k",
        type=int,
        default=None,
        help=(
            "Replan cadence for the _replan family (act every k-th step). "
            "Required for those algorithms, forbidden for the rest."
        ),
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
        help=f"Master seed for the 50-seed derivation (default {DEFAULT_MASTER_SEED}).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help=f"How many episodes to run (default {DEFAULT_NUM_SEEDS}); a prefix of the master stream.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Concurrent subprocesses (default 1 = sequential). N>1 uses a thread pool.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Output directory root; results go in <results-dir>/<world_stem>/<label>/ (label = algorithm, or algorithm_k<K> for replan families).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip seeds whose <seed>.json already exists (default: overwrite).",
    )
    traffic_group = parser.add_mutually_exclusive_group()
    traffic_group.add_argument(
        "--traffic", dest="traffic", action="store_true", help="Enable crossing traffic (default)."
    )
    traffic_group.add_argument(
        "--no-traffic", dest="traffic", action="store_false", help="Disable traffic."
    )
    parser.set_defaults(traffic=True)
    ns = parser.parse_args(argv)
    return RunnerArgs(
        algorithm=ns.algorithm,
        world=ns.world,
        replan_k=None if ns.replan_k is None else int(ns.replan_k),
        master_seed=int(ns.master_seed),
        num_seeds=int(ns.num_seeds),
        jobs=int(ns.jobs),
        results_dir=ns.results_dir,
        resume=bool(ns.resume),
        traffic=bool(ns.traffic),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the batch end-to-end. See module docstring for CLI semantics."""
    args = _parse_args(argv)

    # The CLI guards num_seeds/jobs here (exit 2); derive_episode_seeds() guards num_seeds
    # again as a direct-call safeguard, so its ValueError branch is unreachable from the CLI.
    if args.num_seeds < 1:
        print(f"error: --num-seeds must be >= 1, got {args.num_seeds}", file=sys.stderr)
        return 2
    if args.jobs < 1:
        print(f"error: --jobs must be >= 1, got {args.jobs}", file=sys.stderr)
        return 2
    # A negative master seed would reach np.random.SeedSequence and raise ValueError
    # (a bare traceback) instead of the runner's documented validation-failure path.
    if args.master_seed < 0:
        print(f"error: --master-seed must be >= 0, got {args.master_seed}", file=sys.stderr)
        return 2

    # Validate the (algorithm, replan-k) family combo ONCE before spawning any child
    # (mirrors run_episode's discipline): reject a _replan family with no --replan-k,
    # a --replan-k handed to a non-replan family, or an out-of-range cadence here —
    # exit 2 like the other up-front checks — rather than failing 50 subprocesses.
    try:
        build_controller(args.algorithm, args.replan_k)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Resolve --world to an absolute path ONCE so this process's existence check
    # and the children (launched with cwd=repo_root) agree regardless of cwd.
    world_abs = Path(args.world).resolve()
    if not world_abs.exists():
        print(f"error: --world path does not exist: {world_abs}", file=sys.stderr)
        return 2
    world_abs_str = str(world_abs)
    world_stem = world_abs.stem
    # Manifest records a repo-relative POSIX path when the world lives under the repo, so two
    # developers at the same commit produce identical manifests (falls back to absolute otherwise).
    try:
        world_for_manifest = world_abs.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        world_for_manifest = world_abs_str

    # Resolve the results-dir root to an absolute path ONCE (against this process's
    # cwd) and forward THAT to the children. Children run with cwd=repo_root, so a
    # relative root would otherwise have the parent's manifest/resume probe and the
    # children's episode JSONs land in two different trees.
    results_dir_abs = str(Path(args.results_dir).resolve())

    cpu = os.cpu_count() or 1
    if args.jobs > cpu:
        print(f"note: --jobs {args.jobs} exceeds cpu_count {cpu}; oversubscribing.", file=sys.stderr)

    seeds = derive_episode_seeds(args.master_seed, args.num_seeds)

    # Partition results by the label, not the bare family name, so each child
    # (which computes the same label) lands in this directory and replan cadences
    # do not collide (a_star_replan_k5 vs a_star_replan_k10).
    label = algorithm_label(args.algorithm, args.replan_k)
    out_dir = episode_out_dir(results_dir_abs, world_stem, label)
    out_dir.mkdir(parents=True, exist_ok=True)  # pre-create once; children won't race on it

    print(
        f"batch: algorithm={args.algorithm} world_stem={world_stem} "
        f"master_seed={args.master_seed} num_seeds={args.num_seeds} "
        f"jobs={args.jobs} traffic={args.traffic} resume={args.resume}"
    )

    # Pre-size the result list so we can store each outcome at its derivation index,
    # keeping the manifest order independent of subprocess completion order.
    results: list[EpisodeResult | None] = [None] * len(seeds)

    def _dispatch(index: int) -> EpisodeResult:
        seed = seeds[index]
        # --resume keys off the metrics file existing. run_episode writes <seed>.json only on
        # exit-0 paths (success OR a recorded planner failure); a runner/config fault leaves no
        # file, so resume re-runs those. Resume thus skips "previously ran to completion", not
        # "previously succeeded at the task".
        #
        # Known limitation: the probe trusts that <seed>.json means the prior run finished
        # cleanly. It cannot detect a JSON torn by a kill mid-write, nor a stale JSON whose
        # matching trace is missing — and trace-absence is NOT a usable completeness signal,
        # because a recorded planner failure legitimately writes JSON with no trace. If a prior
        # batch was interrupted, rerun the affected seeds WITHOUT --resume to overwrite cleanly.
        if args.resume and (out_dir / f"{seed}.json").exists():
            return EpisodeResult(seed=seed, exit_code=0, status="skipped", stderr_tail="")
        return _run_one_episode(
            seed=seed,
            algorithm=args.algorithm,
            replan_k=args.replan_k,
            world_abs=world_abs_str,
            results_dir=results_dir_abs,
            traffic=args.traffic,
            repo_root=_REPO_ROOT,
        )

    def _report(index: int, res: EpisodeResult) -> None:
        results[index] = res
        tag = {"ok": "ok ", "skipped": "skip", "runner_error": "FAIL"}[res.status]
        line = f"  [{index + 1}/{len(seeds)}] seed={res.seed} {tag} (exit={res.exit_code})"
        print(line)

    if args.jobs == 1:
        for index in range(len(seeds)):
            _report(index, _dispatch(index))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            future_to_index = {pool.submit(_dispatch, i): i for i in range(len(seeds))}
            for fut in as_completed(future_to_index):
                index = future_to_index[fut]
                _report(index, fut.result())

    ordered = tuple(r for r in results if r is not None)
    summary = summarize(ordered)

    # Provenance receipt — deterministic across same-master-seed runs at the same
    # commit (no timestamp / elapsed; stderr tails are console-only).
    manifest = {
        "master_seed": args.master_seed,
        "num_seeds": args.num_seeds,
        "algorithm": args.algorithm,
        "replan_k": args.replan_k,
        "world": world_for_manifest,
        "world_stem": world_stem,
        "traffic": args.traffic,
        "git_sha": _git_sha(_REPO_ROOT),
        "derived_seeds": list(seeds),
        "episodes": [
            {"seed": r.seed, "exit_code": r.exit_code, "status": r.status} for r in ordered
        ],
    }
    with open(out_dir / MANIFEST_NAME, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, sort_keys=True, indent=2)
        fh.write("\n")

    print(
        f"done: {summary.n_ok} ran to completion, {summary.n_failed} runner-failed, "
        f"{summary.n_skipped} skipped (of {summary.n_total}). manifest: {out_dir / MANIFEST_NAME}"
    )
    if summary.n_failed:
        print("runner failures:")
        for r in ordered:
            if r.status == "runner_error":
                tail = r.stderr_tail.replace("\n", "\n      ")
                print(f"  seed={r.seed} exit={r.exit_code}:\n      {tail}")
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
