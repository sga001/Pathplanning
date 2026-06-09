"""Incremental D* Lite search core in pure CELL space.

This is the hand-rolled incremental replanning engine from Koenig & Likhachev,
"Fast Replanning for Robot Navigation" (the optimised, ``k_m``-based D* Lite).
It operates purely on a boolean occupancy grid in cell coordinates: no
controller, no lidar, no Arena, no world-frame conversion. The controller wiring
that turns this into a live planner is task T11.

Cost model — kept byte-for-byte consistent with ``manual_astar.astar_search``
=============================================================================
The whole point of D* Lite here is to produce the *same optimal paths* a fresh
A* would, so the edge-cost rules below mirror ``astar_search`` exactly:

- 8-connected neighbours, using the same eight deltas in the same order
  (``(-1,-1) (-1,0) (-1,1) (0,-1) (0,1) (1,-1) (1,0) (1,1)``).
- A cell ``c`` is traversable iff it is in bounds AND ``grid_cells[c]`` is False.
  Moving *into* an occupied (or out-of-bounds) cell costs ``inf``.
- Step cost: orthogonal moves cost ``1.0``; diagonal moves cost ``sqrt(2)`` via
  ``np.hypot(delta_row, delta_col)`` — identical to ``astar_search``.
- No corner cutting: a diagonal move from ``u`` to ``v`` is blocked (cost
  ``inf``) if EITHER of the two orthogonally-adjacent shared cells is occupied,
  exactly the ``row_neighbor`` / ``col_neighbor`` check in ``astar_search``.

A* uses a Euclidean heuristic; D* Lite uses the *octile* distance instead. Both
are admissible and consistent for this octile-cost metric, so D* Lite recovers
the same optimal path cost A* does — octile is simply the natural (and tighter)
heuristic for an 8-connected octile-cost grid, and it keeps ``calc_key``
monotone, which D* Lite's correctness proof requires.

Grid ownership
==============
``DStarLiteSearch`` stores a *reference* to the ``grid_cells`` array, NOT a copy.
T11 mutates that array in place (folding live lidar returns onto the static map)
and then tells the search which cells changed via :meth:`update_cells`. Keeping a
reference means the search always reads the caller's current occupancy; the
caller is responsible for reporting every flip through :meth:`update_cells` so
the incremental invariants stay intact.

Determinism
===========
The priority queue is a binary heap of ``(key, counter, cell)`` triples where
``counter`` is a strictly increasing insertion sequence number. Ties on ``key``
are broken by ``counter`` (insertion order), never by cell identity, so no
Python ``set`` iteration or dict ordering can influence which vertex is expanded
first. Given the same ``(grid, start, goal, update-sequence)`` two runs produce
byte-identical paths.
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np

from manual_astar import (
    GRID_RESOLUTION,
    OccupancyGrid,
    SAFETY_MARGIN,
    WAYPOINT_REACHED_DISTANCE,
    WAYPOINT_STRIDE,
    WaypointFollower,
    build_occupancy_grid,
    compute_action_from_state,
    is_cell_in_bounds,
    load_world,
    world_to_grid,
)
from planners._grid import (
    LidarGeometry,
    grid_path_to_waypoints,
    lidar_to_occupancy,
    load_lidar_geometry,
    register,
    segment_is_clear_grid,
)

# The eight 8-connected neighbour deltas, in the SAME order as
# manual_astar.astar_search. Order is load-bearing for deterministic expansion.
NEIGHBOR_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# sqrt(2) - 1, the per-cell penalty octile distance charges for each diagonal
# step beyond the orthogonal baseline.
_OCTILE_DIAGONAL_PENALTY: float = float(np.sqrt(2.0) - 1.0)

# Floating-point tolerance for every key / cost comparison in the engine.
# D* Lite's correctness proof assumes exact arithmetic: it relies on the
# invariant h(s_old, s) + k_m == h(s_new, s) + k_m_old, which holds in the reals
# but NOT in IEEE-754 — the k_m running sum and the octile h are summed in
# different orders at insertion vs recompute, so two mathematically-equal keys
# routinely differ by a rounding ULP (~1e-15). A naive strict `<` then mis-orders
# such a pair and can terminate the search with a still-inconsistent frontier
# vertex queued, manifesting as a phantom "no path" or a sub-optimal path. All
# costs here are sums of 1.0 and sqrt(2) over a small grid, so the smallest gap
# between two genuinely different path costs is many orders of magnitude larger
# than this tolerance; treating differences below it as "equal" is therefore both
# safe and deterministic. Absolute tolerance is sufficient because the magnitudes
# are bounded (a few tens at most for any grid this search is used on).
_KEY_EPSILON: float = 1e-9

Cell = tuple[int, int]
Key = tuple[float, float]


class DStarLiteSearch:
    """Incremental D* Lite shortest-path search over a boolean occupancy grid.

    Coordinates are ``(row, col)`` cell tuples indexing ``grid_cells`` directly,
    matching ``manual_astar.astar_search``'s convention. ``True`` means blocked.

    The search is NOT run in ``__init__``; call :meth:`compute_shortest_path`
    after construction (and again after any :meth:`update_cells` /
    :meth:`move_start`) before reading a path with :meth:`extract_path`.
    """

    def __init__(
        self,
        grid_cells: np.ndarray,
        start_cell: Cell,
        goal_cell: Cell,
    ) -> None:
        if not isinstance(grid_cells, np.ndarray):
            raise ValueError("grid_cells must be a numpy.ndarray.")
        if grid_cells.ndim != 2:
            raise ValueError(
                f"grid_cells must be 2-D (rows, cols); got shape {grid_cells.shape!r}."
            )
        if grid_cells.dtype != np.bool_:
            raise ValueError(
                f"grid_cells must be a boolean array; got dtype {grid_cells.dtype!r}."
            )

        rows, cols = grid_cells.shape
        if rows < 1 or cols < 1:
            raise ValueError(
                f"grid_cells must be non-empty; got shape {grid_cells.shape!r}."
            )

        self._validate_in_bounds(start_cell, rows, cols, label="start_cell")
        self._validate_in_bounds(goal_cell, rows, cols, label="goal_cell")

        # Store a REFERENCE: T11 mutates this array between update_cells calls and
        # the search must observe those mutations. See module docstring.
        self._grid: np.ndarray = grid_cells
        self._rows: int = rows
        self._cols: int = cols

        self._s_start: Cell = (int(start_cell[0]), int(start_cell[1]))
        self._s_goal: Cell = (int(goal_cell[0]), int(goal_cell[1]))
        self._s_last: Cell = self._s_start
        self._k_m: float = 0.0

        # g / rhs default to +inf for every cell (lazily, via .get(..., inf)).
        self._g: dict[Cell, float] = {}
        self._rhs: dict[Cell, float] = {}

        # Lazy-deletion binary heap of (key, counter, cell). Staleness is decided
        # by INSERTION IDENTITY, never by recomputing calc_key: `_latest_counter`
        # records the counter of the most recent _insert for each queued cell, and
        # `_queued` mirrors heap membership. A popped entry is stale (and skipped)
        # iff its cell is no longer queued OR its stored counter no longer matches
        # `_latest_counter[cell]` (a later _insert superseded it). This is what
        # lets already-keyed entries survive move_start: their stored keys stay in
        # the heap and the k_m/start drift is compensated inside the pop loop's
        # re-key branch, NOT by deletion.
        self._heap: list[tuple[Key, int, Cell]] = []
        self._queued: set[Cell] = set()
        self._latest_counter: dict[Cell, int] = {}
        self._counter = itertools.count()

        # Goal is the search root: rhs(goal) = 0, everything else +inf.
        self._rhs[self._s_goal] = 0.0
        self._insert(self._s_goal, self.calc_key(self._s_goal))

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def calc_key(self, s: Cell) -> Key:
        """The D* Lite priority key for cell ``s``.

        ``(min(g, rhs) + h(s_start, s) + k_m, min(g, rhs))`` — the lexicographic
        key that orders the open queue.
        """
        g_rhs_min = min(self._get_g(s), self._get_rhs(s))
        return (g_rhs_min + self._heuristic(self._s_start, s) + self._k_m, g_rhs_min)

    def compute_shortest_path(self) -> None:
        """Run the D* Lite main loop until ``s_start`` is locally consistent.

        Expands the queue while the top key is below ``calc_key(s_start)`` or the
        start is locally inconsistent (``rhs != g``). Overconsistent vertices are
        lowered to their ``rhs``; underconsistent vertices are raised to ``inf``
        and re-evaluated. Stale heap entries (lazy deletions) are discarded.

        The re-key branch is what compensates for the ``k_m`` / start drift
        ``move_start`` accumulates: a queued vertex whose stored key is below its
        live ``calc_key`` is popped and re-inserted with the corrected key BEFORE
        any termination decision, instead of being relaxed prematurely or driving
        the loop guard with an artificially small key. Float robustness is split
        across two mechanisms: this re-key handles a genuinely stale key (the
        first component is meaningfully below the live one after a k_m bump),
        while ``_key_lt`` / ``_floats_equal`` absorb the sub-ULP rounding drift
        the ``h(s_old, s) + k_m == h(s_new, s) + k_m_old`` invariant suffers in
        IEEE-754. Together they guarantee the termination guard only ever compares
        live, rounding-tolerant keys, so a still-inconsistent frontier vertex is
        never skipped and the search never stops one expansion early.
        """
        while True:
            top = self._peek_valid()
            if top is None:
                # Queue exhausted: no inconsistent vertex remains to relax.
                break

            # `k_old` is the entry's STORED key (its key at insertion time), NOT
            # a recomputed calc_key — that is the whole point of Bug #1's fix.
            k_old, u = top
            k_new = self.calc_key(u)

            # A stored key can be badly STALE, not just ULP-off: after move_start
            # bumps k_m, an entry queued long ago carries a first component far
            # below its live value, which would let it sit at the heap top with an
            # artificially small key and drive a premature termination. The
            # canonical optimized D* Lite compensates with the lazy re-key: pop the
            # stale entry, re-insert it with its live key, and re-loop BEFORE any
            # termination decision — so the guard below only ever sees a top whose
            # stored key equals its live key. _key_lt is tolerant to a rounding
            # ULP, so a true k_m/h-drifted-only key is treated as already current.
            if self._key_lt(k_old, k_new):
                self._pop_top()
                self._insert(u, k_new)
                continue

            start_key = self.calc_key(self._s_start)

            # Figure 3 loop guard: stop once the (now non-stale) top key no longer
            # beats the start key AND the start is locally consistent (rhs == g).
            if not self._key_lt(k_old, start_key) and self._floats_equal(
                self._get_rhs(self._s_start), self._get_g(self._s_start)
            ):
                break

            self._pop_top()

            g_u = self._get_g(u)
            rhs_u = self._get_rhs(u)

            if g_u > rhs_u and not self._floats_equal(g_u, rhs_u):
                # Overconsistent: settle g(u) := rhs(u), then UpdateVertex each
                # predecessor (Succ == Pred on this undirected grid).
                self._set_g(u, rhs_u)
                for pred in self._neighbors(u):
                    self._update_vertex(pred)
            else:
                # Underconsistent: raise g(u) := inf, then UpdateVertex u AND each
                # predecessor so their rhs re-propagates around the raised vertex.
                self._set_g(u, np.inf)
                self._update_vertex(u)
                for pred in self._neighbors(u):
                    self._update_vertex(pred)

    def update_cells(self, changed_cells: list[Cell]) -> None:
        """Report occupancy flips so incident edge costs are repaired.

        For each changed cell, every edge touching it changed cost, so we
        re-evaluate the changed cell itself and each of its 8 neighbours via
        ``update_vertex``. The caller must already have mutated ``grid_cells``
        (and, when the robot moved, called :meth:`move_start`) BEFORE calling
        this; here we only fix the affected vertices, then the next
        :meth:`compute_shortest_path` propagates the change.
        """
        if not isinstance(changed_cells, list):
            raise ValueError("changed_cells must be a list of (row, col) tuples.")

        # Collect the changed cells plus their neighbours, de-duplicated but
        # processed in a deterministic order (sorted) so repeated runs match.
        affected: set[Cell] = set()
        for cell in changed_cells:
            normalized = (int(cell[0]), int(cell[1]))
            if not self._in_bounds(normalized):
                raise ValueError(
                    f"changed cell {cell!r} is outside the grid bounds "
                    f"(rows={self._rows}, cols={self._cols})."
                )
            affected.add(normalized)
            for neighbor in self._neighbors(normalized):
                affected.add(neighbor)

        for cell in sorted(affected):
            self._update_vertex(cell)

    def move_start(self, new_start_cell: Cell) -> None:
        """Slide the search start to ``new_start_cell``.

        Accumulates the heuristic drift into ``k_m`` (so previously computed keys
        stay comparable without re-keying the whole queue) and records the move.
        """
        normalized = (int(new_start_cell[0]), int(new_start_cell[1]))
        if not self._in_bounds(normalized):
            raise ValueError(
                f"new_start_cell {new_start_cell!r} is outside the grid bounds "
                f"(rows={self._rows}, cols={self._cols})."
            )

        self._k_m += self._heuristic(self._s_last, normalized)
        self._s_last = normalized
        self._s_start = normalized

    def extract_path(self) -> list[Cell]:
        """Greedily follow the gradient of ``cost(s, s') + g(s')`` to the goal.

        Returns ``[s_start, ..., s_goal]``. Raises ``RuntimeError`` if the start
        is unreachable (``g(s_start)`` is ``inf``) or if a cycle is detected (the
        iteration cap of ``rows * cols`` is exceeded — should never trigger once
        the search is consistent, but guards against a malformed state).
        """
        if not np.isfinite(self._get_g(self._s_start)):
            raise RuntimeError(
                "No path from start to goal: g(s_start) is infinite."
            )

        path: list[Cell] = [self._s_start]
        current = self._s_start
        max_iterations = self._rows * self._cols

        for _ in range(max_iterations):
            if current == self._s_goal:
                return path

            best_successor: Cell | None = None
            best_cost = np.inf
            for successor in self._neighbors(current):
                step = self._cost(current, successor)
                if not np.isfinite(step):
                    continue
                total = step + self._get_g(successor)
                # Strict `<` with deterministic neighbour order means the first
                # minimiser in NEIGHBOR_DELTAS order wins ties — reproducible.
                if total < best_cost:
                    best_cost = total
                    best_successor = successor

            if best_successor is None or not np.isfinite(best_cost):
                raise RuntimeError(
                    "No path from start to goal: gradient descent reached a "
                    "dead end with no finite successor."
                )

            current = best_successor
            path.append(current)

        raise RuntimeError(
            "extract_path exceeded the iteration cap; the search state is "
            "inconsistent or contains a cycle."
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_in_bounds(cell: Cell, rows: int, cols: int, label: str) -> None:
        row, col = int(cell[0]), int(cell[1])
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(
                f"{label} {cell!r} is outside the grid bounds "
                f"(rows={rows}, cols={cols})."
            )

    def _in_bounds(self, cell: Cell) -> bool:
        row, col = cell
        return 0 <= row < self._rows and 0 <= col < self._cols

    def _is_occupied(self, cell: Cell) -> bool:
        # bool() unwraps numpy.bool_ to a native bool for clean comparisons.
        return bool(self._grid[cell])

    def _get_g(self, s: Cell) -> float:
        return self._g.get(s, np.inf)

    def _get_rhs(self, s: Cell) -> float:
        return self._rhs.get(s, np.inf)

    def _set_g(self, s: Cell, value: float) -> None:
        self._g[s] = value

    def _heuristic(self, a: Cell, b: Cell) -> float:
        """Octile distance, consistent with the orthogonal=1 / diagonal=sqrt(2)
        cost model: ``max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)``."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return float(max(dx, dy) + _OCTILE_DIAGONAL_PENALTY * min(dx, dy))

    def _neighbors(self, cell: Cell) -> list[Cell]:
        """In-bounds 8-connected neighbours, in NEIGHBOR_DELTAS order.

        Returns geometric neighbours regardless of occupancy; traversability is
        decided per-edge in :meth:`_cost`. Order is deterministic.
        """
        row, col = cell
        result: list[Cell] = []
        for delta_row, delta_col in NEIGHBOR_DELTAS:
            neighbor = (row + delta_row, col + delta_col)
            if self._in_bounds(neighbor):
                result.append(neighbor)
        return result

    def _cost(self, u: Cell, v: Cell) -> float:
        """Edge cost of moving from ``u`` to an 8-neighbour ``v``.

        Mirrors ``astar_search`` exactly:
        - ``inf`` if ``v`` is out of bounds or occupied (can't move into it);
        - ``inf`` for a diagonal move if EITHER shared orthogonal cell is
          occupied (no corner cutting);
        - otherwise ``np.hypot(delta_row, delta_col)`` (1.0 / sqrt(2)).
        """
        if not self._in_bounds(v) or self._is_occupied(v):
            return np.inf

        delta_row = v[0] - u[0]
        delta_col = v[1] - u[1]

        if delta_row != 0 and delta_col != 0:
            row_neighbor = (u[0] + delta_row, u[1])
            col_neighbor = (u[0], u[1] + delta_col)
            # Both shared orthogonal cells are in bounds here because v is in
            # bounds and they each share one coordinate with v.
            if self._is_occupied(row_neighbor) or self._is_occupied(col_neighbor):
                return np.inf

        return float(np.hypot(delta_row, delta_col))

    def _update_vertex(self, u: Cell) -> None:
        """Recompute ``rhs(u)`` from successors and re-sync ``u``'s queue slot.

        For every cell except the goal, ``rhs(u) = min over successors s of
        cost(u, s) + g(s)``. The vertex is then removed from the queue and, if it
        is locally inconsistent (``g != rhs``), re-inserted with a fresh key.
        """
        if u != self._s_goal:
            best = np.inf
            for successor in self._neighbors(u):
                candidate = self._cost(u, successor) + self._get_g(successor)
                if candidate < best:
                    best = candidate
            self._rhs[u] = best

        # Remove any existing queue slot (lazy: we just drop membership; the
        # stale heap entry is skipped on pop because the cell is no longer queued
        # — or, after a fresh _insert, because its stored counter is superseded).
        self._queued.discard(u)
        self._latest_counter.pop(u, None)

        # Re-queue iff locally inconsistent. The tolerant equality keeps a vertex
        # whose g and rhs differ only by a rounding ULP OUT of the queue, so it is
        # never popped and mis-classified as under/overconsistent in the main
        # loop (which would corrupt g with an inf or a stale rhs).
        if not self._floats_equal(self._get_g(u), self._get_rhs(u)):
            self._insert(u, self.calc_key(u))

    def _insert(self, cell: Cell, key: Key) -> None:
        """Push ``cell`` with ``key``, mark it queued, and record this insertion
        as the cell's latest (so any earlier heap entry for it becomes stale)."""
        counter = next(self._counter)
        heapq.heappush(self._heap, (key, counter, cell))
        self._queued.add(cell)
        self._latest_counter[cell] = counter

    @staticmethod
    def _floats_equal(left: float, right: float) -> bool:
        """``left == right`` within ``_KEY_EPSILON`` (inf-safe).

        ``abs(left - right)`` is ``nan`` for ``inf - inf``, so the explicit
        ``left == right`` short-circuit handles the matching-infinity case (and
        is exact for finite equals); only finite, nearly-equal pairs fall through
        to the tolerance test.
        """
        if left == right:
            return True
        if np.isinf(left) or np.isinf(right):
            return False
        return abs(left - right) <= _KEY_EPSILON

    @classmethod
    def _key_lt(cls, left: Key, right: Key) -> bool:
        """Strict lexicographic ``left < right`` for D* Lite keys, tolerant to a
        floating-point rounding ULP in either component (see ``_KEY_EPSILON``).

        Components within ``_KEY_EPSILON`` are treated as equal so a k_m/h
        rounding drift never inverts the order of two mathematically-equal keys.
        """
        if cls._floats_equal(left[0], right[0]):
            return left[1] < right[1] and not cls._floats_equal(left[1], right[1])
        return left[0] < right[0]

    def _peek_valid(self) -> tuple[Key, Cell] | None:
        """Return the (STORED key, cell) of the smallest *valid* heap entry, None
        if the queue is empty.

        Staleness is decided by INSERTION IDENTITY, never by recomputing
        calc_key (Bug #1): an entry is stale iff its cell is no longer queued OR
        its stored counter no longer matches the cell's latest insertion counter
        (a newer _insert superseded it). Valid entries keep their stored keys so
        the k_m/start drift introduced by move_start is compensated in the pop
        loop's re-key branch rather than by dropping still-needed vertices.
        """
        while self._heap:
            stored_key, stored_counter, cell = self._heap[0]
            if (
                cell not in self._queued
                or self._latest_counter.get(cell) != stored_counter
            ):
                heapq.heappop(self._heap)
                continue
            return stored_key, cell
        return None

    def _pop_top(self) -> None:
        """Pop the validated top entry and clear its queue membership."""
        _, _, cell = heapq.heappop(self._heap)
        self._queued.discard(cell)
        self._latest_counter.pop(cell, None)


# --------------------------------------------------------------------------- #
# Controller wiring (T11)                                                      #
# --------------------------------------------------------------------------- #


class DStarLiteController:
    """Incremental D* Lite `Controller` over a lidar-folded occupancy grid.

    Plans once at `reset()` from the t=0 lidar fold. On every `act()` it re-folds
    the live scan, feeds ONLY the changed cells to the search via
    :meth:`DStarLiteSearch.update_cells`, and lets the search incrementally repair
    its `g`/`rhs` — so the D* Lite machinery runs at every tick, which is the
    point of this family (there is no `_once` / `_replan` split, Mission.md: D*
    Lite is inherently incremental).

    The *path the robot steers by* is re-extracted only when the current waypoint
    follower has finished OR the immediate segment it is about to traverse (the
    robot pose -> its current target waypoint) is no longer clear in the live
    folded grid. Re-extracting on every changed cell instead whipsaws the heading:
    folding a live scan repaints a thick inflation band around every wall return,
    so the optimal cell path jitters one or two cells each tick even on a static
    map, and rebuilding the follower from a jittering path (its index reset to the
    current pose) starves forward speed and the robot times out. Committing to a
    path until its immediate segment is actually blocked keeps the robot at full
    speed on a clear run yet still replans the instant a dynamic obstacle crosses
    in front of it.

    Grid ownership is the load-bearing invariant: `self._cells` is the SAME
    ndarray the search was constructed with. `act()` mutates that array in place
    at the flipped positions and reports them through `update_cells`; it never
    rebinds `self._cells` to a freshly folded array (that would detach the
    search's view and silently desynchronise the incremental edge costs).

    A t=0 planning failure in `reset()` propagates so the runner records
    `planner_error`. A mid-episode replan failure in `act()` is swallowed — the
    last valid follower is kept, never rebuilt — so `act()` never raises (AC8).
    """

    name = "d_star_lite"

    def __init__(self, replan_k: int | None = None) -> None:
        # `build_controller` rejects `--replan-k` for d_star_lite (it is not in
        # REPLAN_FAMILIES) before construction; the kwarg is accepted here only to
        # match the uniform `ALGORITHMS[name](replan_k=...)` seam, then ignored.
        del replan_k

        self._grid: OccupancyGrid | None = None
        self._static_cells: np.ndarray | None = None
        self._cells: np.ndarray | None = None
        self._geom: LidarGeometry | None = None
        self._inflation: float = 0.0
        self._goal_xy: np.ndarray | None = None
        self._search: DStarLiteSearch | None = None
        self._follower: WaypointFollower | None = None

    def reset(
        self,
        world_yaml: str,
        initial_snapshot: tuple,
        lidar0: np.ndarray,
        state0: np.ndarray,
    ) -> None:
        # `initial_snapshot` is ignored by design: this family is lidar-only and
        # `lidar0` already encodes the t=0 obstacles.
        del initial_snapshot

        world = load_world(world_yaml)
        grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)

        goal_xy = np.asarray(world.goal, dtype=float)
        goal_cell = world_to_grid(goal_xy, grid)
        if not is_cell_in_bounds(goal_cell, grid):
            raise ValueError("The goal position is outside the occupancy grid.")
        if bool(grid.cells[goal_cell]):
            raise ValueError("The goal position is blocked after obstacle inflation.")

        self._grid = grid
        self._static_cells = grid.cells
        self._geom = load_lidar_geometry(world_yaml)
        self._inflation = world.robot_radius + SAFETY_MARGIN
        self._goal_xy = goal_xy

        start_cell = world_to_grid(state0[:2], grid)

        # The MUTABLE working occupancy the search holds a reference to. From here
        # on it is mutated IN PLACE by act(); it is never rebound.
        self._cells = lidar_to_occupancy(
            self._static_cells, grid, state0, lidar0, self._geom, self._inflation
        )

        self._search = DStarLiteSearch(self._cells, start_cell, goal_cell)
        self._search.compute_shortest_path()
        # Raises RuntimeError if the start cannot reach the goal; the runner turns
        # that into planner_error.
        cells_path = self._search.extract_path()

        waypoints = grid_path_to_waypoints(
            cells_path, grid, self._cells, state0[:2], self._goal_xy, WAYPOINT_STRIDE
        )
        if not waypoints:
            raise ValueError("The initial plan produced no waypoints.")

        self._follower = WaypointFollower(list(waypoints), WAYPOINT_REACHED_DISTANCE)

    def act(self, state: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        if (
            self._search is None
            or self._follower is None
            or self._grid is None
            or self._static_cells is None
            or self._cells is None
            or self._geom is None
            or self._goal_xy is None
        ):
            raise RuntimeError("act() called before reset().")

        position = np.asarray(state[:2], dtype=float)

        new_cells = lidar_to_occupancy(
            self._static_cells, self._grid, state, lidar, self._geom, self._inflation
        )
        diff_mask = self._cells != new_cells

        if bool(diff_mask.any()):
            # Deterministic, de-duplicated list of the flipped (row, col) cells.
            changed = sorted(
                (int(row), int(col)) for row, col in zip(*np.where(diff_mask))
            )

            # CRITICAL: mutate the EXISTING array in place — the search holds a
            # reference to it. Do NOT rebind self._cells to new_cells, which would
            # detach the search's view and desynchronise its incremental costs.
            self._cells[diff_mask] = new_cells[diff_mask]

            current_cell = world_to_grid(position, self._grid)
            # Canonical optimized D* Lite order: bump k_m via move_start FIRST, then
            # repair the changed edges, then recompute the shortest-path tree. This
            # runs every changed tick so the incremental g/rhs stays current; the
            # path the robot follows is only re-extracted on demand below.
            self._search.move_start(current_cell)
            self._search.update_cells(changed)
            try:
                self._search.compute_shortest_path()
            except (ValueError, RuntimeError):
                # Keep the last valid follower; never rebuild it (AC8). A failed
                # incremental pass leaves the previous g/rhs in place, so the held
                # path stays drivable.
                return compute_action_from_state(state, self._follower)

        # Re-extract the followed path only when the held one is exhausted or its
        # imminent segment is now blocked — otherwise keep committing to it so a
        # clear run holds full speed instead of chasing the lidar-fold jitter.
        if self._follower.is_finished or self._immediate_segment_blocked(position):
            try:
                cells_path = self._search.extract_path()
                waypoints = grid_path_to_waypoints(
                    cells_path,
                    self._grid,
                    self._cells,
                    position,
                    self._goal_xy,
                    WAYPOINT_STRIDE,
                )
                if waypoints:
                    self._follower = WaypointFollower(
                        list(waypoints), WAYPOINT_REACHED_DISTANCE
                    )
            except (ValueError, RuntimeError):
                # Keep the last valid path/follower; never rebuild it (AC8).
                pass

        return compute_action_from_state(state, self._follower)

    def _immediate_segment_blocked(self, position: np.ndarray) -> bool:
        """Is the segment the robot is about to traverse no longer clear?

        The commitment horizon is exactly the robot pose -> its current target
        waypoint: the one piece of the held path the robot will drive across next.
        Checking only this segment (against the live folded grid, so it sees both
        static walls and dynamic traffic) reacts promptly to an obstacle that
        actually crosses the robot's path, while ignoring the fold's far-field
        inflation noise that would otherwise force a replan every tick.
        """
        assert self._follower is not None and self._grid is not None
        assert self._cells is not None
        target = self._follower.current_waypoint(position)
        return not segment_is_clear_grid(self._cells, self._grid, position, target)


register("d_star_lite", DStarLiteController)
