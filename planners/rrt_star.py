"""RRT* controllers: single-shot (``rrt_star_once``) and periodic-replan (``rrt_star_replan``).

RRT* is RRT plus two optimality steps after every node insertion — *choose-parent*
(connect the new node through whichever nearby node minimizes its cost-to-come) and
*rewire* (re-route nearby nodes through the new node when that is cheaper). Those
steps give the tree its asymptotic-optimality property: the reconstructed path is
observably shorter than plain RRT's for the same sample sequence.

This module REUSES the deterministic RRT core from ``planners.rrt`` verbatim — the
goal-biased sampler, the steer function, the allocation-free ``_segment_clear_fast``
edge check, the buffer-backed ``_nearest_index_in_array`` nearest-node search, the
parent-walk reconstruction, ``rrt_points_to_waypoints``, ``rrt_planned_cost``, and
the shared tuned constants (``RRT_SEED``, ``RRT_MAX_ITERS``, ``RRT_STEP``,
``RRT_GOAL_BIAS``, ``RRT_GOAL_TOLERANCE``, ``POINT_EPSILON``). Importing these
underscore-prefixed helpers across modules within the ``planners`` package is
intentional reuse (the plan: "Imports the sampling/steer/nearest/collision helpers
from planners/rrt.py"): it keeps the sample sequence and node positions byte-for-byte
identical to RRT, so only the parent structure (and thus the rewired, shorter path)
differs.

``_segment_clear_fast`` and ``_nearest_index_in_array`` are the T3/T4 perf twins of
``segment_is_clear_grid`` and the per-iteration ``np.asarray(nodes)`` rebuild: each
returns the BIT-IDENTICAL bool / index over the same inputs, so swapping them in keeps
the planned path, cost, and ``trace.jsonl`` byte-identical while removing the inner
loop's per-edge numpy boxing and the growing node-buffer rebuild.

**Why determinism survives the optimality steps.** In RRT* a new node's POSITION is
still ``_steer(nearest, sample)`` exactly as in plain RRT — choose-parent and rewire
only change parent pointers and per-node cost-to-come, never node positions. So with
the SAME ``RRT_SEED`` the sample sequence and the node positions are IDENTICAL to RRT;
only the tree's parent structure differs. That is why ``rrt_star_once`` connects to the
goal at the same iteration RRT does (AC5/AC7 hold with seed 5) while producing a
shorter path (AC7-obs). The NEAR set is gathered in ascending node-index order, ties in
choose-parent break to the first minimum, and rewire iterates the NEAR set ascending, so
no set/dict iteration order leaks in — two same-seed ``rrt_star_once`` runs produce
byte-identical traces (AC4).

``rrt_star_plan`` shares ``rrt_plan``'s signature and return contract (continuous
start-first/goal-last point list, ``ValueError`` on no-path) so the controllers below
call it the same way the RRT controllers call ``rrt_plan``.
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
    load_world,
)
from planners._grid import (
    PathFollowingController,
    register,
)
from planners._types import Path
from planners.rrt import (
    POINT_EPSILON,
    RRT_GOAL_TOLERANCE,
    RRT_MAX_ITERS,
    RRT_SEED,
    RRT_STEP,
    _cell_is_free,
    _nearest_index_in_array,
    _reconstruct_points,
    _sample_free_point,
    _segment_clear_fast,
    _steer,
    rrt_points_to_waypoints,
)

# --- Tunable constants ------------------------------------------------------

# Neighborhood radius (m) for the choose-parent + rewire NEAR set. Set to several
# steer steps so each new node has a non-trivial set of candidate re-parents/rewire
# targets (a radius below RRT_STEP would leave the NEAR set near-empty and collapse
# RRT* back to plain RRT). 3 m == 3 * RRT_STEP: wide enough to find a cheaper parent
# across the staircase a grid path would take, narrow enough to keep each insertion's
# collision-check work bounded. The new node's POSITION is unchanged by this radius
# (it is always _steer(nearest, sample)), so widening/narrowing it never perturbs the
# sample sequence — only the parent structure and thus the planned cost.
RRT_STAR_NEIGHBOR_RADIUS = 3.0

# Measured: rrt_star_once on arena_v1.yaml --no-traffic with RRT_SEED=5 reaches the
# goal in 70.7 sim-seconds (recorded by T5 verification, seed 42), well under the 120 s
# cap and the ~110 s target. AC7-obs planned-cost observation at seed 5 (computed
# in-process): rrt_once planned cost 78.00 m vs rrt_star_once 70.92 m — rewiring shaved
# ~7.08 m off the planned path (path nodes dropped 79 -> 35) while connecting to the
# goal at the same iteration RRT does.


def rrt_star_plan(
    grid_cells: np.ndarray,
    grid: OccupancyGrid,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Grow an asymptotically-optimal RRT* from ``start_xy`` to ``goal_xy``.

    Same signature and return contract as ``planners.rrt.rrt_plan`` (continuous
    start-first/goal-last point list; raises ``ValueError`` on cap exhaustion or a
    blocked/out-of-bounds start or goal), so the controllers call it identically.

    Each iteration: sample free space (goal-biased, via ``_sample_free_point``), find
    the nearest tree node, steer toward the sample by ``RRT_STEP`` to get ``p_new``,
    and skip the iteration when the ``nearest -> p_new`` edge is not collision-free.
    Then:

    - **choose-parent:** among the NEAR set (existing nodes within
      ``RRT_STAR_NEIGHBOR_RADIUS`` of ``p_new``, iterated in ascending index order)
      whose edge to ``p_new`` is collision-free, pick the parent minimizing
      ``cost[near] + dist(near, p_new)`` (first minimum on ties). When no NEAR node is
      collision-reachable, fall back to ``nearest`` (the standard RRT edge, already
      known clear).
    - **rewire:** for each ``near`` in the NEAR set (ascending), if routing it through
      the new node is strictly cheaper AND that edge is collision-free, re-parent it
      and propagate the cost delta to its descendants.

    The node POSITIONS are byte-identical to ``rrt_plan`` for the same ``rng`` — only
    parent pointers and costs change — which is what keeps the trace deterministic
    (AC4) while shortening the reconstructed path (AC7-obs).
    """
    start_point = np.asarray(start_xy, dtype=float)
    goal_point = np.asarray(goal_xy, dtype=float)

    if not _cell_is_free(start_point, grid_cells, grid):
        raise ValueError("RRT* start position is blocked or outside the grid.")
    if not _cell_is_free(goal_point, grid_cells, grid):
        raise ValueError("RRT* goal position is blocked or outside the grid.")

    # Parallel lists: nodes[i] is a world-frame point, parents[i] the index of its
    # parent (-1 for the root), costs[i] the cost-to-come from the start. `children`
    # is the inverse of `parents`, maintained so rewire can propagate a cost delta to
    # a re-parented node's whole subtree in deterministic (ascending-index) order.
    # These lists are KEPT (not replaced by the buffer): `_choose_parent`, `_rewire`,
    # `_reconstruct_points`, and the cost/children bookkeeping all index `nodes[i]`.
    nodes: list[np.ndarray] = [start_point]
    parents: list[int] = [-1]
    costs: list[float] = [0.0]
    children: list[list[int]] = [[]]

    # Incremental node-position buffer (Part B): a preallocated, C-contiguous
    # `(capacity, 2)` float array mirroring `nodes`, scanned as a `[:count]` slice for
    # the nearest-neighbour argmin AND the NEAR-set radius query. This replaces the
    # per-iteration `np.asarray(nodes)` rebuild that ran inside both `_nearest_node_index`
    # and `_near_node_indices` — small now (~4-7%) but growing proportionally once the
    # LOS check is fast — while staying byte-identical to it (same float values, same
    # C-contiguous layout, same einsum reductions, same argmin/flatnonzero tie order).
    # Doubling growth keeps appends amortized O(1). The buffer is appended at the SAME
    # point the node is accepted into `nodes` (after choose-parent assigns the index),
    # so `positions[:count]` always holds exactly the nodes that existed when nearest
    # and near were queried for this iteration.
    capacity = 16
    positions = np.empty((capacity, 2), dtype=float)
    positions[0] = start_point
    count = 1

    for _ in range(RRT_MAX_ITERS):
        sample = _sample_free_point(grid, goal_point, rng)
        nearest_index = _nearest_index_in_array(positions, count, sample)
        nearest_point = nodes[nearest_index]
        new_point = _steer(nearest_point, sample)

        # The nearest edge is the fallback parent; if even it is blocked the steered
        # node is unreachable, so drop the iteration (mirrors rrt_plan's skip).
        if not _segment_clear_fast(grid_cells, grid, nearest_point, new_point):
            continue

        near_indices = _near_node_indices(positions, count, new_point)

        parent_index, parent_cost = _choose_parent(
            nodes, costs, near_indices, nearest_index, new_point, grid_cells, grid
        )

        new_index = len(nodes)
        nodes.append(new_point)
        parents.append(parent_index)
        costs.append(parent_cost)
        children.append([])
        children[parent_index].append(new_index)

        # Mirror the accepted node into the position buffer, doubling capacity when
        # full so the contiguous scan slice always covers every tree node. This happens
        # after the index is assigned so positions[:count] matches `nodes` exactly.
        if count == capacity:
            capacity *= 2
            grown = np.empty((capacity, 2), dtype=float)
            grown[:count] = positions[:count]
            positions = grown
        positions[count] = new_point
        count += 1

        _rewire(
            nodes, parents, costs, children, near_indices, new_index, grid_cells, grid
        )

        if float(np.linalg.norm(new_point - goal_point)) <= RRT_GOAL_TOLERANCE:
            # Reconstruct through the rewired tree: walking parents back from the
            # goal-connected node yields the optimized (re-parented) path, not the raw
            # insertion-order chain (this is where RRT*'s shorter path materializes).
            return _reconstruct_points(nodes, parents, new_index, goal_point)

    raise ValueError(
        f"RRT* exhausted {RRT_MAX_ITERS} iterations without reaching the goal."
    )


def _near_node_indices(
    positions: np.ndarray, count: int, new_point: np.ndarray
) -> list[int]:
    """Indices of existing tree nodes within ``RRT_STAR_NEIGHBOR_RADIUS`` of ``new_point``.

    Returned in ascending index order (the natural scan order of the node list) so the
    downstream choose-parent / rewire passes are fully order-deterministic — no
    distance sort, no set, so ties never reorder run to run (AC4).

    Buffer-backed (Part B): scans the contiguous ``positions[:count]`` slice of the
    preallocated ``(capacity, 2)`` float buffer instead of rebuilding
    ``np.asarray(nodes)`` every iteration. This is BYTE-IDENTICAL to the prior
    list-based version — same float values (the buffer mirrors ``nodes`` exactly),
    same ``einsum`` reduction, same ``radius_sq`` (``RRT_STAR_NEIGHBOR_RADIUS``
    squared), and ``np.flatnonzero`` already scans ascending — so the NEAR set, and
    thus the chosen parents / rewired costs / planned path, are unchanged (the
    byte-identity is load-bearing for the trace). The function is module-private (used
    only by ``rrt_star_plan``), so the buffer-typed signature is safe.
    """
    deltas = positions[:count] - new_point
    distances_sq = np.einsum("ij,ij->i", deltas, deltas)
    radius_sq = RRT_STAR_NEIGHBOR_RADIUS * RRT_STAR_NEIGHBOR_RADIUS
    # np.flatnonzero scans ascending, so the result is already index-ordered.
    return [int(index) for index in np.flatnonzero(distances_sq <= radius_sq)]


def _choose_parent(
    nodes: list[np.ndarray],
    costs: list[float],
    near_indices: list[int],
    nearest_index: int,
    new_point: np.ndarray,
    grid_cells: np.ndarray,
    grid: OccupancyGrid,
) -> tuple[int, float]:
    """Pick the cheapest collision-free parent for ``new_point`` from the NEAR set.

    Among NEAR nodes whose edge to ``new_point`` is collision-free, returns the one
    minimizing ``costs[near] + dist(near, new_point)`` (first minimum on ties, since
    the scan is strictly ``<`` over the ascending ``near_indices``). When the NEAR set
    yields no collision-free candidate it falls back to ``nearest_index`` — whose edge
    ``rrt_star_plan`` has already confirmed clear — so a parent is always returned.

    Returns ``(parent_index, cost_to_come_of_new_point)``.
    """
    # Fallback: the standard RRT edge through the nearest node (already clear).
    best_index = nearest_index
    best_cost = costs[nearest_index] + float(
        np.linalg.norm(new_point - nodes[nearest_index])
    )

    for near_index in near_indices:
        if near_index == nearest_index:
            # Already accounted for as the fallback; its edge is clear by construction.
            continue
        near_point = nodes[near_index]
        if not _segment_clear_fast(grid_cells, grid, near_point, new_point):
            continue
        candidate_cost = costs[near_index] + float(np.linalg.norm(new_point - near_point))
        if candidate_cost < best_cost - POINT_EPSILON:
            best_cost = candidate_cost
            best_index = near_index

    return best_index, best_cost


def _rewire(
    nodes: list[np.ndarray],
    parents: list[int],
    costs: list[float],
    children: list[list[int]],
    near_indices: list[int],
    new_index: int,
    grid_cells: np.ndarray,
    grid: OccupancyGrid,
) -> None:
    """Re-route NEAR nodes through the new node when that lowers their cost-to-come.

    For each ``near`` in ``near_indices`` (ascending), if
    ``costs[new] + dist(new, near) < costs[near]`` AND the ``new -> near`` edge is
    collision-free, detach ``near`` from its current parent, re-parent it onto
    ``new_index``, update its cost, and propagate the resulting cost delta to its whole
    subtree via ``_propagate_cost``. Iterating ascending and using a strict
    ``< - POINT_EPSILON`` improvement test keeps the rewiring byte-deterministic (AC4).
    The new node is never rewired onto itself, and re-parenting onto a descendant is
    impossible because the new node is a leaf with no descendants when this runs.
    """
    new_point = nodes[new_index]
    new_cost = costs[new_index]

    for near_index in near_indices:
        if near_index == new_index:
            continue
        near_point = nodes[near_index]
        rerouted_cost = new_cost + float(np.linalg.norm(near_point - new_point))
        if rerouted_cost >= costs[near_index] - POINT_EPSILON:
            continue
        if not _segment_clear_fast(grid_cells, grid, new_point, near_point):
            continue

        # Detach `near` from its old parent and graft it under the new node.
        old_parent = parents[near_index]
        if old_parent != -1:
            children[old_parent].remove(near_index)
        parents[near_index] = new_index
        children[new_index].append(near_index)

        cost_delta = rerouted_cost - costs[near_index]
        costs[near_index] = rerouted_cost
        _propagate_cost(children, costs, near_index, cost_delta)


def _propagate_cost(
    children: list[list[int]],
    costs: list[float],
    root_index: int,
    cost_delta: float,
) -> None:
    """Add ``cost_delta`` to every descendant of ``root_index`` (cost-to-come shift).

    When a node is rewired, the same additive change to its cost-to-come applies to its
    entire subtree (each descendant's cost is its ancestor chain length, which shifted
    by exactly ``cost_delta``). An explicit ascending-order stack walk keeps this
    deterministic and avoids Python recursion-depth limits on a deep tree; the root's
    own cost is already updated by the caller, so only its descendants are touched here.
    """
    stack = list(children[root_index])
    while stack:
        node_index = stack.pop()
        costs[node_index] += cost_delta
        stack.extend(children[node_index])


class RRTStarOnceController:
    """Single-shot RRT*: plan once at t=0 on the static map, then follow forever.

    Mirrors ``RRTOnceController`` exactly (a standalone ``Controller``, NOT a
    ``PathFollowingController``): it plans on the STATIC occupancy grid with no lidar
    fold, so its result lives on the same substrate A*/RRT use (keeps AC5/AC7 an
    apples-to-apples comparison). The plan is deterministic from the fixed ``RRT_SEED``;
    ``act()`` ignores the live lidar and just drives the follower. The difference from
    RRT is solely the planner call — ``rrt_star_plan`` rewires the tree for a shorter
    path while connecting to the goal at the same iteration RRT does.
    """

    name = "rrt_star_once"

    def __init__(self, replan_k: int | None = None) -> None:
        # `build_controller` rejects a non-None `replan_k` for the _once family before
        # construction; the kwarg is accepted here only to match the uniform
        # `ALGORITHMS[name](replan_k=...)` construction seam, then ignored.
        del replan_k
        self._follower: WaypointFollower | None = None

    def reset(
        self,
        world_yaml: str,
        initial_snapshot: tuple,
        lidar0: np.ndarray,
        state0: np.ndarray,
    ) -> None:
        # The static plan is fully determined by the world YAML + the fixed seed; the
        # live snapshot and t=0 lidar carry no information this planner uses. state0 is
        # accepted to match the interface but the WORLD start is used so the plan runs
        # on exactly the substrate A*/Dijkstra/RRT use (AC5).
        del initial_snapshot, lidar0, state0

        world = load_world(world_yaml)
        grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)

        start_xy = np.asarray(world.start, dtype=float)[:2]
        goal_xy = np.asarray(world.goal, dtype=float)[:2]

        # Deterministic single-plan RNG (AC4); propagates ValueError on no-path so the
        # runner can record planner_error.
        rng = np.random.default_rng(RRT_SEED)
        points = rrt_star_plan(grid.cells, grid, start_xy, goal_xy, rng)

        waypoints = rrt_points_to_waypoints(
            points, grid, grid.cells, start_xy, goal_xy, WAYPOINT_STRIDE
        )
        if not waypoints:
            raise ValueError("The initial RRT* plan produced no waypoints.")

        self._follower = WaypointFollower(list(waypoints), WAYPOINT_REACHED_DISTANCE)

    def act(self, state: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        if self._follower is None:
            raise RuntimeError("act() called before reset().")

        del lidar  # single-shot follower ignores live lidar
        return compute_action_from_state(state, self._follower)


class RRTStarReplanController(PathFollowingController):
    """Periodic-replan RRT*: re-grow the rewired tree on the lidar-folded grid every K acts.

    Subclasses ``PathFollowingController`` and overrides ONLY ``_plan``, inheriting the
    base's fold + commitment-horizon machinery (identical to ``RRTReplanController``,
    differing only in the ``rrt_star_plan`` call). Each replan uses
    ``default_rng(RRT_SEED + self._k)`` so successive replans explore DIFFERENT samples
    (a replan re-deriving the same path would defeat replanning) while staying
    byte-deterministic — the ``_k`` sequence is itself deterministic.
    """

    name = "rrt_star_replan"

    def _plan(self, folded_grid: OccupancyGrid, folded: np.ndarray, state: np.ndarray) -> Path:
        rng = np.random.default_rng(RRT_SEED + self._k)  # per-plan, deterministic
        start_xy = np.asarray(state[:2], dtype=float)
        points = rrt_star_plan(folded, folded_grid, start_xy, self._goal_xy, rng)  # raises on no-path
        return rrt_points_to_waypoints(
            points, self._grid, folded, start_xy, self._goal_xy, WAYPOINT_STRIDE
        )


register("rrt_star_once", RRTStarOnceController)
register("rrt_star_replan", RRTStarReplanController)
