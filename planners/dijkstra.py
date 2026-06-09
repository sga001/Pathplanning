"""Dijkstra controllers: single-shot (dijkstra_once) and periodic-replan (dijkstra_replan).

Dijkstra is A* with a zero heuristic, so these reuse the exact same search and grid
machinery the A* controllers use; only the heuristic differs.

`DijkstraOnceController` deliberately plans on the STATIC occupancy grid (not the lidar
fold), exactly like `AStarOnceController`, so its path is computed on the same substrate
as A*'s and AC5's "cost equals A*'s" holds. It is therefore a standalone `Controller`,
NOT a `PathFollowingController` (whose pipeline folds lidar).

`DijkstraReplanController` IS a `PathFollowingController` (it replans on the lidar fold,
like `a_star_replan`), differing from A* only by the zero heuristic.
"""
from __future__ import annotations

import numpy as np

from manual_astar import (
    GRID_RESOLUTION,
    SAFETY_MARGIN,
    WAYPOINT_REACHED_DISTANCE,
    WAYPOINT_STRIDE,
    OccupancyGrid,
    WaypointFollower,
    WorldModel,
    astar_search,
    build_occupancy_grid,
    compute_action_from_state,
    load_world,
    path_to_waypoints,
    validate_start_and_goal,
)
from planners._grid import PathFollowingController, register


def _dijkstra_static_waypoints(
    world_yaml: str,
) -> tuple[WorldModel, OccupancyGrid, list[np.ndarray]]:
    """Static-grid Dijkstra: the `plan_waypoints` pipeline with a zero heuristic.

    Mirrors `manual_astar.plan_waypoints` step for step but runs `astar_search`
    with `heuristic_fn=lambda *_: 0.0` (literal zeros => uniform-cost / Dijkstra)
    instead of the default Euclidean heuristic. Propagates the same
    ValueError/RuntimeError the A* pipeline raises so the runner can record
    `planner_error`.
    """
    world = load_world(world_yaml)
    grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)
    start_cell, goal_cell = validate_start_and_goal(world, grid)
    path = astar_search(grid, start_cell, goal_cell, lambda *_: 0.0)
    waypoints = path_to_waypoints(path, world, grid, WAYPOINT_STRIDE)
    return world, grid, waypoints


class DijkstraOnceController:
    """Single-shot Dijkstra: plan once at t=0 on the static map, then follow forever.

    Mirrors `AStarOnceController` exactly; the only difference is the zero
    heuristic inside `_dijkstra_static_waypoints`. Like that controller, it is a
    standalone `Controller` (NOT a `PathFollowingController`) so it plans on the
    static occupancy grid rather than the lidar fold.
    """

    name = "dijkstra_once"

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
        # The static plan is fully determined by the world YAML; the live
        # snapshot, lidar, and start pose carry no information Dijkstra uses here.
        del initial_snapshot, lidar0, state0

        _, _, waypoints = _dijkstra_static_waypoints(world_yaml)
        if not waypoints:
            raise ValueError("The initial plan produced no waypoints.")

        self._follower = WaypointFollower(list(waypoints), WAYPOINT_REACHED_DISTANCE)

    def act(self, state: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        if self._follower is None:
            raise RuntimeError("act() called before reset().")

        del lidar  # single-shot follower ignores live lidar
        return compute_action_from_state(state, self._follower)


class DijkstraReplanController(PathFollowingController):
    """Periodic-replan Dijkstra: re-search the lidar-folded grid every `replan_k` acts.

    `heuristic_fn = staticmethod(lambda *_: 0.0)` makes the base `compute_path`
    feed `astar_search` a heuristic of literal zeros, turning the A* search into
    a uniform-cost (Dijkstra) search. Accessing the `staticmethod` via the class
    (`type(self).heuristic_fn`) yields the plain callable, which `astar_search`
    treats as non-None and uses directly.
    """

    name = "dijkstra_replan"
    heuristic_fn = staticmethod(lambda *_: 0.0)


register("dijkstra_once", DijkstraOnceController)
register("dijkstra_replan", DijkstraReplanController)
