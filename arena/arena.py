from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TIMEOUT_S = 120.0
LIDAR_BEAM_COUNT = 360
ACTION_SHAPE = (2, 1)


class ArenaConfigError(ValueError):
    """Raised at Arena.__init__ for malformed config (e.g. lidar beam count mismatch)."""


class ArenaRuntimeError(RuntimeError):
    """Raised mid-episode for irsim contract violations (e.g. lidar dict missing 'ranges')."""


@dataclass(frozen=True)
class EpisodeInfo:
    sim_time: float
    step_idx: int
    crashed: bool
    timed_out: bool
    reached_goal: bool
    distance_to_goal: float
    wallclock_per_step: float
    dynamic_obstacle_count: int
    lidar_status: str


class Arena:
    """Static 50x50 arena wrapping irsim. Phase 0 = no dynamic obstacles."""

    def __init__(
        self,
        yaml_path: str | Path,
        seed: int,
        render: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        raise NotImplementedError("T3")

    def reset(self) -> tuple[np.ndarray, np.ndarray, EpisodeInfo]:
        raise NotImplementedError("T3")

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, EpisodeInfo]:
        raise NotImplementedError("T3")

    @property
    def initial_dynamic_snapshot(self) -> tuple[Any, ...]:
        """Snapshot of dynamic obstacles at t=0. Empty in Phase 0; Phase 2 narrows the type."""
        return ()

    def close(self) -> None:
        raise NotImplementedError("T3")
