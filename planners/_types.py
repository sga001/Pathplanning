from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np

Path = tuple[np.ndarray, ...]  # ordered (2,)-shaped float64 world-frame waypoints; last == goal


@runtime_checkable
class PathPlanner(Protocol):
    name: str  # e.g. "a_star_once" — must match the results/<name>/ subdir

    def plan(
        self,
        world_yaml: str,
        initial_dynamic_snapshot: tuple,  # () in Phase 1 / Phase 0; tuple of states in Phase 2
        lidar0: np.ndarray,               # (360,) float64 from arena.reset(); may be all-NaN if reset's lidar_status == "missing". Static planners (A*) ignore it; reactive planners (Phase 6) use it.
    ) -> Path: ...
