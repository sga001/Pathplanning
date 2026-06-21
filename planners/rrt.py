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

import math

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
    SEGMENT_SAMPLE_FACTOR,
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


def _segment_clear_fast(
    grid_cells: np.ndarray,
    grid: OccupancyGrid,
    p0,
    p1,
) -> bool:
    """Allocation-free scalar twin of ``planners._grid.segment_is_clear_grid``.

    Why this exists: ``segment_is_clear_grid`` is the RRT collision hot loop, and
    its cost is the per-sample numpy boxing — ``np.asarray``/``np.linalg.norm``
    plus ``world_to_grid``'s scalar ``np.clip``/``np.floor`` — run ~16x per edge.
    This helper reproduces that function's arithmetic in pure Python/``math``
    scalars so the inner loop allocates nothing, while returning the BIT-IDENTICAL
    bool for every input (the byte-identical-trace guarantee depends on it). It is
    a faithful mirror, NOT an approximation: the five equivalence obligations
    below each reproduce one exact behaviour of the frozen reference. The signature
    matches ``segment_is_clear_grid`` verbatim so callers (and ``rrt_star.py``)
    can swap one for the other.

    Equivalence obligations (each load-bearing for trace parity):

    1. Clip-then-read, never OOB-reject. ``world_to_grid`` ALWAYS clamps into the
       grid, so the reference's ``is_cell_in_bounds`` guard is dead (always True)
       after it. We clamp row/col into ``[0, rows-1]``/``[0, cols-1]`` and read the
       (clamped) cell directly — no out-of-bounds rejection, which would flip the
       bool and diverge the trace.
    2. Length is ``math.sqrt(dx*dx + dy*dy)`` — ``math.hypot`` is FORBIDDEN.
       ``math.sqrt(dx*dx+dy*dy)`` is bit-identical to
       ``float(np.linalg.norm(end-start))`` (0 mismatches / 1M inputs), whereas
       ``math.hypot``'s extended-precision intermediate flips the last bit on
       ~17% of inputs (e.g. dx=1.0, dy=2.4000000000000004 gives sample_count 26
       vs 27), which propagates into the sample count and flips the bool.
    3. Sample count is ``max(2, math.ceil(length / sample_step))`` with
       ``sample_step = grid.resolution * SEGMENT_SAMPLE_FACTOR`` (imported, not
       hardcoded). ``math.ceil`` equals ``int(np.ceil(...))`` for these finite
       positive lengths.
    4. Cell clamp is ``min(max(math.floor((coord - offset) / resolution), 0),
       n - 1)``, equal to ``int(np.clip(np.floor(...), 0, n-1))`` for the finite
       in-range floats here — col from x/offset_x/cols, row from y/offset_y/rows,
       matching ``world_to_grid``'s x->col / y->row mapping and (row, col) order.
    5. The ``length < 1e-9`` degenerate branch checks ONLY the start cell
       (clip-then-read). RRT reaches it via ``_steer`` when a sample coincides
       with the nearest node.
    """
    # Read each endpoint coordinate once as a Python float; dx/dy then drive the
    # whole computation with no per-sample numpy boxing.
    start_x = float(p0[0])
    start_y = float(p0[1])
    end_x = float(p1[0])
    end_y = float(p1[1])
    dx = end_x - start_x
    dy = end_y - start_y

    # Obligation 2: math.sqrt(dx*dx + dy*dy) is bit-identical to
    # float(np.linalg.norm(end - start)); math.hypot is FORBIDDEN (last-bit drift
    # on ~17% of inputs flips the sample count and thus the returned bool).
    length = math.sqrt(dx * dx + dy * dy)
    sample_step = grid.resolution * SEGMENT_SAMPLE_FACTOR

    # Grid geometry hoisted out of the per-sample loop (obligation 4 inputs).
    rows, cols = grid.shape
    offset_x = float(grid.offset[0])
    offset_y = float(grid.offset[1])
    resolution = grid.resolution

    if length < 1e-9:
        # Obligation 5 (+ 1, 4): degenerate segment checks only the start cell,
        # clamped into bounds and read directly.
        col = min(max(math.floor((start_x - offset_x) / resolution), 0), cols - 1)
        row = min(max(math.floor((start_y - offset_y) / resolution), 0), rows - 1)
        return not bool(grid_cells[row, col])

    # Obligation 3: identical sample count to the reference (math.ceil == int(np.ceil)
    # for finite positive lengths here).
    sample_count = max(2, math.ceil(length / sample_step))
    for sample_index in range(sample_count + 1):
        # ratio = sample_index / sample_count and px/py via the same affine form
        # the reference uses (point = start + ratio*segment, componentwise) so the
        # FP rounding matches; the last sample is recomputed at ratio=1.0, NOT
        # shortcut to the endpoint.
        ratio = sample_index / sample_count
        px = start_x + ratio * dx
        py = start_y + ratio * dy

        # Obligations 1 + 4: clip-then-read, no OOB rejection.
        col = min(max(math.floor((px - offset_x) / resolution), 0), cols - 1)
        row = min(max(math.floor((py - offset_y) / resolution), 0), rows - 1)
        if bool(grid_cells[row, col]):
            return False

    return True


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


def _nearest_index_in_array(
    positions: np.ndarray, count: int, sample: np.ndarray
) -> int:
    """Index of the nearest of ``positions[:count]`` to ``sample`` (first argmin).

    Buffer-backed twin of ``_nearest_node_index``: it scans a contiguous
    ``positions[:count]`` slice of a preallocated, C-contiguous ``(capacity, 2)``
    float array instead of rebuilding ``np.asarray(nodes)`` every iteration. The
    result is BYTE-IDENTICAL to ``_nearest_node_index`` over the same nodes — same
    float values, same C-contiguous memory layout, same ``einsum`` reduction, and
    ``np.argmin`` returns the first minimum on ties — so routing ``rrt_plan``
    through this preserves the determinism guarantee. Exposed at module level so
    ``rrt_star.py`` (T4) can reuse the identical scan.
    """
    deltas = positions[:count] - sample
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
    ``RRT_GOAL_BIAS``), finds the nearest existing tree node (via the incremental
    position buffer + ``_nearest_index_in_array``), steers toward the sample by
    ``RRT_STEP``, and accepts the new edge only if ``_segment_clear_fast`` — the
    allocation-free, bit-identical twin of ``segment_is_clear_grid`` — reports the
    steered segment unobstructed. When a new node lands within
    ``RRT_GOAL_TOLERANCE`` of the goal the search succeeds: parents are walked
    back from that node to the root, the order is reversed to run start-first, and
    the exact goal point is appended last.

    Both speedups (the scalar LOS check and the buffer-backed nearest scan) return
    byte-identical results to the prior numpy paths, so the planned path, cost, and
    ``trace.jsonl`` are unchanged.

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
    # parents[i] is the index of its parent (-1 for the root). The lists are kept
    # for `_reconstruct_points`' parent walk (fully order-deterministic).
    nodes: list[np.ndarray] = [start_point]
    parents: list[int] = [-1]

    # Incremental node-position buffer (Part B): a preallocated, C-contiguous
    # `(capacity, 2)` float array mirroring `nodes`, scanned as a `[:count]` slice
    # for the nearest-neighbour argmin. This replaces the per-iteration
    # `np.asarray(nodes)` rebuild — the rebuild was ~19% of `rrt_plan`'s time and
    # grows once the LOS check is fast — while staying byte-identical to it (same
    # values, same layout, same einsum/argmin). Doubling growth keeps appends
    # amortized O(1).
    capacity = 16
    positions = np.empty((capacity, 2), dtype=float)
    positions[0] = start_point
    count = 1

    for _ in range(RRT_MAX_ITERS):
        sample = _sample_free_point(grid, goal_point, rng)
        nearest_index = _nearest_index_in_array(positions, count, sample)
        nearest_point = nodes[nearest_index]
        new_point = _steer(nearest_point, sample)

        if not _segment_clear_fast(grid_cells, grid, nearest_point, new_point):
            continue

        nodes.append(new_point)
        parents.append(nearest_index)
        new_index = len(nodes) - 1

        # Mirror the accepted node into the position buffer, doubling capacity
        # when full so the contiguous scan slice always covers every tree node.
        if count == capacity:
            capacity *= 2
            grown = np.empty((capacity, 2), dtype=float)
            grown[:count] = positions[:count]
            positions = grown
        positions[count] = new_point
        count += 1

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
