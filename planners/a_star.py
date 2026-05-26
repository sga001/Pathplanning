"""A* path planner adapter — wraps manual_astar.plan_waypoints behind the PathPlanner Protocol."""
from __future__ import annotations

import numpy as np

from manual_astar import plan_waypoints
from planners._types import Path


class AStarOncePlanner:
    """Single-shot A* planner: plans once at t=0 against the static map."""

    name = "a_star_once"

    def plan(
        self,
        world_yaml: str,
        initial_dynamic_snapshot: tuple,  # noqa: ARG002 — Phase 2 hook, unused by A*
        lidar0: np.ndarray,                # noqa: ARG002 — reactive-planner hook, unused by A*
    ) -> Path:
        _, _, _, waypoints = plan_waypoints(world_yaml)
        return tuple(waypoints)
