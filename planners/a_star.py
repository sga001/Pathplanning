"""A* controllers: single-shot (a_star_once) and periodic-replan (a_star_replan)."""
from __future__ import annotations

import numpy as np

from manual_astar import (
    WAYPOINT_REACHED_DISTANCE,
    WaypointFollower,
    compute_action_from_state,
    plan_waypoints,
)
from planners._grid import PathFollowingController, register


class AStarOnceController:
    """Single-shot A*: plan once at t=0 on the static map, then follow forever.

    Deliberately a standalone `Controller` (NOT a `PathFollowingController`) so it
    keeps using the analytic line-of-sight `plan_waypoints` pipeline from
    `manual_astar`. This keeps it byte-identical to the prior `AStarOncePlanner`
    so its recorded traces do not move (AC3).
    """

    name = "a_star_once"

    def __init__(self, replan_k: int | None = None) -> None:
        # `build_controller` rejects a non-None `replan_k` for the _once family
        # before construction; the kwarg is accepted here only to match the
        # uniform `ALGORITHMS[name](replan_k=...)` construction seam, then ignored.
        del replan_k
        self._follower: WaypointFollower | None = None

    def reset(
        self,
        world_yaml: str,
        initial_snapshot: tuple,
        lidar0: np.ndarray,
        state0: np.ndarray,
    ) -> None:
        # The static plan is fully determined by the world YAML; the live snapshot,
        # lidar, and start pose carry no information A* uses here.
        del initial_snapshot, lidar0, state0

        # Analytic A* on the static map; propagates ValueError/RuntimeError so the
        # runner can record planner_error.
        _, _, _, waypoints = plan_waypoints(world_yaml)
        if not waypoints:
            raise ValueError("The initial plan produced no waypoints.")

        self._follower = WaypointFollower(list(waypoints), WAYPOINT_REACHED_DISTANCE)

    def act(self, state: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        if self._follower is None:
            raise RuntimeError("act() called before reset().")

        del lidar  # single-shot follower ignores live lidar
        return compute_action_from_state(state, self._follower)


class AStarReplanController(PathFollowingController):
    """Periodic-replan A*: re-search the lidar-folded grid every `replan_k` acts.

    `heuristic_fn = None` selects the Euclidean-heuristic A* path inside
    `astar_search` (the base `compute_path` reads `type(self).heuristic_fn`).
    """

    name = "a_star_replan"
    heuristic_fn = None


register("a_star_once", AStarOnceController)
register("a_star_replan", AStarReplanController)
