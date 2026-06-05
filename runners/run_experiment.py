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
        [--master-seed <int>]   # default DEFAULT_MASTER_SEED
        [--num-seeds <int>]     # default 50; >= 1
        [--jobs <int>]          # default 1 (sequential); N>1 => bounded concurrency
        [--results-dir <dir>]   # default "results"; forwarded to each episode
        [--resume]              # skip seeds whose <seed>.json already exists
        [--traffic|--no-traffic]# Phase 2 crossing traffic, default ON

Outputs (per seed, written by the child run_episode):
    <results-dir>/<world_stem>/<algorithm>/<seed>.json
    <results-dir>/<world_stem>/<algorithm>/<seed>.trace.jsonl   (only on planning success)
Plus a provenance receipt written by this module:
    <results-dir>/<world_stem>/<algorithm>/_manifest.json

Execution:
    --jobs 1 (default) runs seeds sequentially. --jobs N>1 runs up to N child
    subprocesses concurrently via a ThreadPoolExecutor (threads waiting on
    subprocess.run — no multiprocessing, so the Windows spawn/pickle path never
    enters). Result bytes are identical at any --jobs value because each seed is
    isolated; only `wallclock_per_step` (a Mission.md "freebie") is perturbed by
    contention. Produce headline wallclock numbers with --jobs 1.

Exit codes:
    0 — every non-skipped seed's subprocess exited 0 (ran to completion)
    1 — >= 1 seed's subprocess exited non-zero (continue-and-report)
    2 — argparse error / up-front validation failure (unknown algorithm, missing world)

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

from runners.run_episode import ALGORITHMS  # noqa: E402


DEFAULT_MASTER_SEED = 20260605          # canonical experiment stream; value is arbitrary
DEFAULT_NUM_SEEDS = 50                   # Mission.md: 50 seeds per algorithm
DEFAULT_RESULTS_DIR = "results"
MANIFEST_NAME = "_manifest.json"         # underscore prefix; Phase 5 globs episodes by numeric stem
STDERR_TAIL_LINES = 20                   # last N lines of a failed child's stderr, for the console
EPISODE_TIMEOUT_S = 600.0                # hard wall on one child; the in-sim cap is 120 s sim-time,
                                         # so ~10 min wallclock means the child is wedged, not slow
TIMEOUT_EXIT_CODE = 124                  # synthetic exit code recorded for a timed-out child (mirrors GNU timeout)


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
        )
    except (OSError, ValueError):
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
    cmd.append("--traffic" if traffic else "--no-traffic")
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
        help="Output directory root; results go in <results-dir>/<world_stem>/<algorithm>/.",
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

    cpu = os.cpu_count() or 1
    if args.jobs > cpu:
        print(f"note: --jobs {args.jobs} exceeds cpu_count {cpu}; oversubscribing.", file=sys.stderr)

    seeds = derive_episode_seeds(args.master_seed, args.num_seeds)

    out_dir = Path(args.results_dir) / world_stem / args.algorithm
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
        # "previously succeeded at the task". It assumes the prior run wrote files cleanly.
        if args.resume and (out_dir / f"{seed}.json").exists():
            return EpisodeResult(seed=seed, exit_code=0, status="skipped", stderr_tail="")
        return _run_one_episode(
            seed=seed,
            algorithm=args.algorithm,
            world_abs=world_abs_str,
            results_dir=args.results_dir,
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
