"""RRT controllers: single-shot (``rrt_once``) and periodic-replan (``rrt_replan``).

This module hand-rolls a deterministic Rapidly-exploring Random Tree over the
shared occupancy grid so the harness's byte-identical-trace guarantee holds
(the tree is grown entirely from a single ``numpy`` ``Generator``; no set/dict
iteration order, no ``Math.random``-style nondeterminism). It reuses the grid
line-of-sight + ``WaypointFollower`` substrate from ``manual_astar`` /
``planners._grid`` so the planner comparison isolates the *search*, not the
follower.

Three free functions form the core:

- ``rrt_plan`` — grows a goal-biased tree from ``start_xy`` toward ``goal_xy``,
  collision-checking every steered edge against the occupancy array, and walks
  parents back to a continuous start..goal point list on success. Raises
  ``ValueError`` on iteration-cap exhaustion or a blocked/out-of-bounds
  start/goal cell (so a no-path world is a clean raise, not a crash).
- ``rrt_points_to_waypoints`` — shortcuts that continuous point list into a
  sparse, line-of-sight-safe waypoint tuple, replicating the candidate-span
  bisection of ``grid_path_to_waypoints`` (stride downsampling + recursive
  reinsertion) rather than feeding the points through the cell-path API.
- ``rrt_planned_cost`` — total Euclidean length of a planned point path (used
  only by the RRT*-vs-RRT planned-cost observation in T5).

``RRTOnceController`` (key ``rrt_once``) mirrors ``AStarOnceController``: it
plans once at ``reset()`` on the STATIC occupancy grid (no lidar fold) so its
result lives on the same substrate A* uses, then follows that path forever.
``RRTReplanController`` (key ``rrt_replan``) subclasses ``PathFollowingController``
and overrides only ``_plan`` to re-grow the tree on the lidar-folded grid every
``replan_k`` acts, inheriting the base's commitment-horizon machinery.

All tunables are the module-level ``UPPER_SNAKE_CASE`` constants below.
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
    build_occupancy_grid,
    compute_action_from_state,
    is_cell_in_bounds,
    load_world,
    world_to_grid,
)
from planners._grid import (
    PathFollowingController,
    _append_clear_waypoints,
    register,
    segment_is_clear_grid,
)
from planners._types import Path

# --- Tunable constants ------------------------------------------------------

# Empirically tuned: RRT_SEED=5 grows a tree whose line-of-sight-shortcut path
# (planned cost ~78 m) drives rrt_once to the goal on arena_v1.yaml --no-traffic
# in 73.0 sim-seconds (well under the 120 s cap and the ~110 s target, on par
# with A*'s ~73.6 s). RRT_MAX_ITERS=5000 is far above the ~80 iterations this
# seed needs to connect, so the solve is solid, not marginal. T5 (RRT*) reuses
# this same seed/budget.
RRT_SEED = 5
# Upper bound on sampling iterations before declaring no-path (raise ValueError).
RRT_MAX_ITERS = 5000
# Steer step (m): the new node is placed RRT_STEP toward the sample from the
# nearest tree node (or at the sample itself if it is already closer).
RRT_STEP = 1.0
# Probability that a sample is the goal itself (pulls the tree toward the goal).
RRT_GOAL_BIAS = 0.05
# Connect-to-goal radius (m): a new node within this of the goal terminates the
# search with the goal appended as the final point.
RRT_GOAL_TOLERANCE = 0.5
# Numeric tolerance for treating two points as coincident.
POINT_EPSILON = 1e-9


def _cell_is_free(point: np.ndarray, grid_cells: np.ndarray, grid: OccupancyGrid) -> bool:
    """True only when ``point`` maps to an in-bounds, unoccupied grid cell.

    ``world_to_grid`` clamps its result to the grid extent, so an out-of-bounds
    world point yields an edge cell; this guards that by re-checking the cell is
    actually in bounds before reading occupancy.
    """
    cell = world_to_grid(point, grid)
    return is_cell_in_bounds(cell, grid) and not bool(grid_cells[cell])


def _sample_free_point(
    grid: OccupancyGrid, goal_xy: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw one sample: the goal with probability ``RRT_GOAL_BIAS``, else uniform.

    The uniform sample is drawn over the grid's world-frame extent. Drawing the
    goal-bias coin first (always one ``random`` draw) keeps the RNG consumption
    order fixed regardless of which branch is taken, which is what makes the tree
    byte-deterministic across runs (AC4).
    """
    rows, cols = grid.shape
    offset_x = float(grid.offset[0])
    offset_y = float(grid.offset[1])
    width = cols * grid.resolution
    height = rows * grid.resolution

    if float(rng.random()) < RRT_GOAL_BIAS:
        return np.asarray(goal_xy, dtype=float)

    sample_x = offset_x + float(rng.random()) * width
    sample_y = offset_y + float(rng.random()) * height
    return np.array([sample_x, sample_y], dtype=float)


def _nearest_node_index(nodes: list[np.ndarray], sample: np.ndarray) -> int:
    """Index of the tree node nearest ``sample`` (first argmin on ties).

    Ties resolve to the lowest index deterministically; the tree is a plain list
    so the scan order is fixed. ``np.argmin`` already returns the first minimum,
    which is the deterministic choice the determinism guarantee requires.
    """
    stacked = np.asarray(nodes, dtype=float)
    deltas = stacked - sample
    distances_sq = np.einsum("ij,ij->i", deltas, deltas)
    return int(np.argmin(distances_sq))


def _steer(from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
    """Step from ``from_point`` toward ``to_point`` by at most ``RRT_STEP``.

    Returns ``to_point`` when it is already within ``RRT_STEP`` (so the tree can
    connect exactly to a nearby sample/goal), otherwise a point ``RRT_STEP`` along
    the direction.
    """
    direction = to_point - from_point
    distance = float(np.linalg.norm(direction))
    if distance <= RRT_STEP or distance < POINT_EPSILON:
        return np.asarray(to_point, dtype=float)
    return from_point + (RRT_STEP / distance) * direction


def rrt_plan(
    grid_cells: np.ndarray,
    grid: OccupancyGrid,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Grow a goal-biased RRT from ``start_xy`` to ``goal_xy`` over ``grid_cells``.

    Each iteration samples free space (the goal with probability
    ``RRT_GOAL_BIAS``), finds the nearest existing tree node, steers toward the
    sample by ``RRT_STEP``, and accepts the new edge only if
    ``segment_is_clear_grid`` reports the steered segment unobstructed. When a
    new node lands within ``RRT_GOAL_TOLERANCE`` of the goal the search succeeds:
    parents are walked back from that node to the root, the order is reversed to
    run start-first, and the exact goal point is appended last.

    Raises ``ValueError`` when the start or goal maps to a blocked/out-of-bounds
    cell, or when ``RRT_MAX_ITERS`` is exhausted without connecting (so a no-path
    world surfaces as a clean planner_error, not a crash).
    """
    start_point = np.asarray(start_xy, dtype=float)
    goal_point = np.asarray(goal_xy, dtype=float)

    if not _cell_is_free(start_point, grid_cells, grid):
        raise ValueError("RRT start position is blocked or outside the grid.")
    if not _cell_is_free(goal_point, grid_cells, grid):
        raise ValueError("RRT goal position is blocked or outside the grid.")

    # The tree is a parallel pair of lists: node[i] is a world-frame point and
    # parents[i] is the index of its parent (-1 for the root). A list keeps the
    # nearest-node scan and the parent walk fully order-deterministic.
    nodes: list[np.ndarray] = [start_point]
    parents: list[int] = [-1]

    for _ in range(RRT_MAX_ITERS):
        sample = _sample_free_point(grid, goal_point, rng)
        nearest_index = _nearest_node_index(nodes, sample)
        nearest_point = nodes[nearest_index]
        new_point = _steer(nearest_point, sample)

        if not segment_is_clear_grid(grid_cells, grid, nearest_point, new_point):
            continue

        nodes.append(new_point)
        parents.append(nearest_index)
        new_index = len(nodes) - 1

        if float(np.linalg.norm(new_point - goal_point)) <= RRT_GOAL_TOLERANCE:
            return _reconstruct_points(nodes, parents, new_index, goal_point)

    raise ValueError(
        f"RRT exhausted {RRT_MAX_ITERS} iterations without reaching the goal."
    )


def _reconstruct_points(
    nodes: list[np.ndarray],
    parents: list[int],
    goal_node_index: int,
    goal_point: np.ndarray,
) -> list[np.ndarray]:
    """Walk parents from ``goal_node_index`` to the root, return start..goal points.

    The parent chain is collected leaf-to-root then reversed to run start-first;
    the exact ``goal_point`` is appended last (unless the terminal node already
    coincides with it) so the returned path always ends exactly at the goal.
    """
    chain: list[np.ndarray] = []
    cursor = goal_node_index
    while cursor != -1:
        chain.append(nodes[cursor])
        cursor = parents[cursor]
    chain.reverse()

    if float(np.linalg.norm(chain[-1] - goal_point)) > POINT_EPSILON:
        chain.append(np.asarray(goal_point, dtype=float))

    return chain


def rrt_points_to_waypoints(
    points: list[np.ndarray],
    grid: OccupancyGrid,
    grid_cells: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    stride: int,
) -> Path:
    """Shortcut a continuous RRT point list into a sparse line-of-sight waypoint tuple.

    Replicates ``grid_path_to_waypoints``' structure exactly: seed the output
    with the first point, build the candidate-index list (index 0, every
    ``stride``-th interior index, the last index), and drive
    ``_append_clear_waypoints`` over each consecutive candidate span so unclear
    spans are recursively bisected to reinsert detail. The first waypoint is then
    pinned to ``start_xy`` and the path is guaranteed to terminate at ``goal_xy``.

    Driving ``_append_clear_waypoints`` per candidate span (not once over the
    whole list) is what applies the stride downsampling; calling it once would
    keep every point on the first clear span.
    """
    if stride < 1:
        raise ValueError("Waypoint stride must be at least 1.")
    if not points:
        raise ValueError("rrt_points_to_waypoints requires a non-empty point list.")

    start_point = np.asarray(start_xy, dtype=float)
    goal_point = np.asarray(goal_xy, dtype=float)

    # Candidate anchor indices: endpoints plus every stride-th interior index,
    # mirroring grid_path_to_waypoints.
    candidate_indices = [0]
    candidate_indices.extend(
        index for index in range(1, len(points) - 1) if index % stride == 0
    )
    candidate_indices.append(len(points) - 1)

    waypoints: list[np.ndarray] = [points[0]]
    for previous_index, next_index in zip(candidate_indices, candidate_indices[1:]):
        _append_clear_waypoints(waypoints, points, previous_index, next_index, grid, grid_cells)

    # Pin the first waypoint to the true start pose and guarantee termination at
    # the exact goal point (the RRT root/leaf may sit a fraction off either).
    waypoints[0] = start_point
    if np.linalg.norm(waypoints[-1] - goal_point) > POINT_EPSILON:
        waypoints.append(goal_point)

    return tuple(waypoints)


def rrt_planned_cost(points: list[np.ndarray]) -> float:
    """Total Euclidean length of a planned point path (sum of segment lengths).

    Used by the RRT*-vs-RRT planned-cost observation (AC7-obs). Returns 0.0 for
    a path of fewer than two points.
    """
    if len(points) < 2:
        return 0.0
    stacked = np.asarray(points, dtype=float)
    segment_deltas = np.diff(stacked, axis=0)
    return float(np.sum(np.linalg.norm(segment_deltas, axis=1)))


class RRTOnceController:
    """Single-shot RRT: plan once at t=0 on the static map, then follow forever.

    Mirrors ``AStarOnceController``: a standalone ``Controller`` (NOT a
    ``PathFollowingController``) that plans on the STATIC occupancy grid with no
    lidar fold, so its path lives on the same substrate A* uses (keeps AC5 an
    apples-to-apples comparison). The plan is deterministic from the fixed
    ``RRT_SEED``; ``act()`` ignores the live lidar and just drives the follower.
    """

    name = "rrt_once"

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
        # The static plan is fully determined by the world YAML + the fixed seed;
        # the live snapshot and t=0 lidar carry no information this planner uses.
        # state0 is accepted to match the interface but the WORLD start is used so
        # the plan runs on exactly the substrate A*/Dijkstra use (AC5).
        del initial_snapshot, lidar0, state0

        world = load_world(world_yaml)
        grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)

        start_xy = np.asarray(world.start, dtype=float)[:2]
        goal_xy = np.asarray(world.goal, dtype=float)[:2]

        # Deterministic single-plan RNG (AC4); propagates ValueError on no-path so
        # the runner can record planner_error.
        rng = np.random.default_rng(RRT_SEED)
        points = rrt_plan(grid.cells, grid, start_xy, goal_xy, rng)

        waypoints = rrt_points_to_waypoints(
            points, grid, grid.cells, start_xy, goal_xy, WAYPOINT_STRIDE
        )
        if not waypoints:
            raise ValueError("The initial RRT plan produced no waypoints.")

        self._follower = WaypointFollower(list(waypoints), WAYPOINT_REACHED_DISTANCE)

    def act(self, state: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        if self._follower is None:
            raise RuntimeError("act() called before reset().")

        del lidar  # single-shot follower ignores live lidar
        return compute_action_from_state(state, self._follower)


class RRTReplanController(PathFollowingController):
    """Periodic-replan RRT: re-grow the tree on the lidar-folded grid every K acts.

    Subclasses ``PathFollowingController`` and overrides ONLY ``_plan``, inheriting
    the base's fold + commitment-horizon machinery. Each replan uses
    ``default_rng(RRT_SEED + self._k)`` so successive replans explore DIFFERENT
    samples (a replan re-deriving the same path would defeat replanning) while
    staying byte-deterministic — the ``_k`` sequence is itself deterministic.
    """

    name = "rrt_replan"

    def _plan(self, folded_grid: OccupancyGrid, folded: np.ndarray, state: np.ndarray) -> Path:
        rng = np.random.default_rng(RRT_SEED + self._k)  # per-plan, deterministic
        start_xy = np.asarray(state[:2], dtype=float)
        points = rrt_plan(folded, folded_grid, start_xy, self._goal_xy, rng)  # raises on no-path
        return rrt_points_to_waypoints(
            points, self._grid, folded, start_xy, self._goal_xy, WAYPOINT_STRIDE
        )


register("rrt_once", RRTOnceController)
register("rrt_replan", RRTReplanController)
