"""Incremental D* Lite search core in pure CELL space, on a flat padded grid.

This is the hand-rolled incremental replanning engine from Koenig & Likhachev,
"Fast Replanning for Robot Navigation" (the optimised, ``k_m``-based D* Lite).
It operates purely on a boolean occupancy grid in cell coordinates: no
controller, no lidar, no Arena, no world-frame conversion. The controller
wiring that turns this into a live planner is :class:`DStarLiteController`.

Internal layout — flat, padded, list-based
===========================================
The public surface speaks ``(row, col)`` tuples, but the hot path works on a
**flat, padded** representation that eliminates the per-edge Python-function and
``numpy`` scalar overhead that dominated the original dict/tuple core:

- **Padding.** A single ring of permanently-occupied border cells surrounds the
  original ``rows x cols`` grid. The padded width is ``W = cols + 2``; the flat
  index of an original ``(row, col)`` is ``(row + 1) * W + (col + 1)``. Because
  the border is always occupied, *moving into* an out-of-bounds cell costs
  ``inf`` for free — no per-edge bounds check is needed in the hot path. An
  ``_interior`` mask (a ``bytearray``) flags the real cells, so predecessor
  loops skip enqueuing border vertices (mirroring the old ``_neighbors``
  in-bounds filter, so no wasted heap churn on the padding).
- **Occupancy mirror.** ``self._grid`` keeps the live ``ndarray`` reference (the
  ownership contract below is unchanged), but a fast padded ``list[bool]`` mirror
  (``self._occ``) is built once in ``__init__`` and re-synced ONLY inside
  :meth:`update_cells`, re-reading each reported changed cell's live value. Every
  hot-path occupancy read hits the list mirror, not the ndarray. This makes the
  *report-every-flip* contract load-bearing for occupancy correctness, not just
  for the incremental invariants: a flip the caller never reports is never
  re-synced into the mirror and the search will not see it.
- **g / rhs as flat Python lists** of native floats sized ``n_padded`` (border
  entries included but never relaxed), initialised to ``inf``. No dicts, no
  ``numpy`` scalars in the hot path — costs are summed with ``math.sqrt`` and the
  module-level :data:`INF`.
- **Per-delta edge metadata** (:data:`_EDGE_META`) is precomputed once at import
  in NEIGHBOR_DELTAS order, each entry ``(offset, step_cost, is_diagonal,
  ortho_a, ortho_b)`` where the diagonal's two no-corner-cut neighbours are the
  flat offsets ``dr*W`` and ``dc``. Orthogonal entries carry no ortho offsets.
- **Heap entries are flat 4-tuples** ``(key0, key1, counter, cell_idx)`` — the
  SAME comparison order as the old ``((k0, k1), counter, cell)``. Lazy deletion
  is by insertion identity: ``self._latest[idx]`` records the counter of the most
  recent insert for ``idx`` (0 = not queued); a popped entry is stale iff its
  stored counter no longer matches ``_latest[idx]``. ``_update_vertex`` marks a
  dequeue with ``_latest[idx] = 0``.

Cost model — kept byte-for-byte consistent with ``manual_astar.astar_search``
=============================================================================
The whole point of D* Lite here is to produce the *same optimal paths* a fresh
A* would, so the edge-cost rules below mirror ``astar_search`` exactly:

- 8-connected neighbours, using the same eight deltas in the same order
  (``(-1,-1) (-1,0) (-1,1) (0,-1) (0,1) (1,-1) (1,0) (1,1)``).
- A cell ``c`` is traversable iff it is in bounds AND unoccupied. Moving *into*
  an occupied (or out-of-bounds, i.e. padding) cell costs ``inf``.
- Step cost: orthogonal moves cost ``1.0``; diagonal moves cost ``sqrt(2)`` —
  identical to ``astar_search``'s ``np.hypot(delta_row, delta_col)``.
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
:class:`DStarLiteController` mutates that array in place (folding live lidar
returns onto the static map) and then tells the search which cells changed via
:meth:`update_cells`. Keeping a reference means the search always reads the
caller's current occupancy when it re-syncs the mirror; the caller is
responsible for reporting every flip through :meth:`update_cells` so both the
occupancy mirror and the incremental invariants stay intact.

Determinism
===========
The priority queue is a binary heap of ``(key0, key1, counter, cell_idx)``
4-tuples where ``counter`` is a strictly increasing insertion sequence number.
Ties on the ``(key0, key1)`` pair are broken by ``counter`` (insertion order),
never by cell identity, so no Python ``set`` iteration or dict ordering can
influence which vertex is expanded first. The affected-vertex set in
:meth:`update_cells` is processed in sorted flat-index order (flat-index sort ==
``(row, col)`` lexicographic sort), and ``extract_path`` breaks ties by first
minimiser in NEIGHBOR_DELTAS order. Given the same
``(grid, start, goal, update-sequence)`` two runs produce byte-identical paths.
"""

from __future__ import annotations

import heapq
import math

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
# manual_astar.astar_search. Order is load-bearing for deterministic expansion
# and for the first-minimiser tie-break in extract_path. Kept as a module
# constant for documentation/order reference; the hot path uses the flat
# per-delta metadata derived from it (see _build_edge_meta).
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

# Native-float infinity; used everywhere so no numpy scalar enters the hot path.
INF: float = float("inf")

# Orthogonal / diagonal step costs as native floats.
_STEP_ORTHO: float = 1.0
_STEP_DIAG: float = math.sqrt(2.0)

# sqrt(2) - 1, the per-cell penalty octile distance charges for each diagonal
# step beyond the orthogonal baseline.
_OCTILE_DIAGONAL_PENALTY: float = math.sqrt(2.0) - 1.0

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


def _build_edge_meta(
    width: int,
) -> tuple[tuple[int, float, bool, int, int], ...]:
    """Precompute the per-delta edge metadata in NEIGHBOR_DELTAS order.

    Each entry is ``(offset, step_cost, is_diagonal, ortho_a, ortho_b)`` where
    ``offset`` is the flat-index delta of the neighbour on a padded grid of width
    ``width``. For a diagonal ``(dr, dc)`` the two cells that must both be free to
    avoid corner-cutting are ``u + dr*W`` (the row-neighbour) and ``u + dc`` (the
    col-neighbour); those flat offsets are stored as ``ortho_a`` / ``ortho_b``.
    Orthogonal entries set ``is_diagonal=False`` and leave the ortho offsets at 0
    (never read).
    """
    meta: list[tuple[int, float, bool, int, int]] = []
    for delta_row, delta_col in NEIGHBOR_DELTAS:
        offset = delta_row * width + delta_col
        is_diagonal = delta_row != 0 and delta_col != 0
        if is_diagonal:
            ortho_a = delta_row * width  # row-neighbour: (u_row + dr, u_col)
            ortho_b = delta_col          # col-neighbour: (u_row, u_col + dc)
            meta.append((offset, _STEP_DIAG, True, ortho_a, ortho_b))
        else:
            meta.append((offset, _STEP_ORTHO, False, 0, 0))
    return tuple(meta)


class DStarLiteSearch:
    """Incremental D* Lite shortest-path search over a boolean occupancy grid.

    The public coordinates are ``(row, col)`` cell tuples indexing ``grid_cells``
    directly, matching ``manual_astar.astar_search``'s convention (``True`` means
    blocked). Internally the search runs on a flat, padded representation (see the
    module docstring): a one-cell occupied border, flat ``g`` / ``rhs`` lists, a
    padded occupancy mirror, and 4-tuple heap entries.

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

        # Store a REFERENCE: DStarLiteController mutates this array between
        # update_cells calls and the search must observe those mutations when it
        # re-syncs the occupancy mirror. See module docstring ("Grid ownership").
        self._grid: np.ndarray = grid_cells
        self._rows: int = rows
        self._cols: int = cols

        # Padded geometry. One ring of permanently-occupied border cells: the
        # padded grid is (rows + 2) x (cols + 2), stored flat in row-major order.
        self._width: int = cols + 2
        self._height: int = rows + 2
        self._n: int = self._width * self._height
        self._edge_meta = _build_edge_meta(self._width)

        # Occupancy mirror: a padded list[bool] with the border set True. Built
        # once here, re-synced only at the reported flips inside update_cells.
        padded = np.pad(grid_cells, pad_width=1, mode="constant", constant_values=True)
        self._occ: list[bool] = padded.ravel().tolist()

        # Interior mask: 1 for real (non-border) cells, 0 for the border ring.
        # Used to skip enqueuing border vertices in the predecessor loops.
        interior = np.zeros((self._height, self._width), dtype=np.uint8)
        interior[1 : rows + 1, 1 : cols + 1] = 1
        self._interior: bytearray = bytearray(interior.ravel().tobytes())

        # Flat start / goal as padded indices, plus the original (row, col) the
        # bookkeeping (calc_key, move_start, extract_path output) reasons about.
        self._s_start_idx: int = self._to_padded(int(start_cell[0]), int(start_cell[1]))
        self._s_goal_idx: int = self._to_padded(int(goal_cell[0]), int(goal_cell[1]))
        self._s_last_idx: int = self._s_start_idx
        self._k_m: float = 0.0

        # g / rhs as flat native-float lists sized n_padded. Border entries are
        # present but never relaxed (the interior mask gates re-enqueue).
        self._g: list[float] = [INF] * self._n
        self._rhs: list[float] = [INF] * self._n

        # Lazy-deletion binary heap of (key0, key1, counter, cell_idx). Staleness
        # is decided by INSERTION IDENTITY, never by recomputing calc_key:
        # `_latest[idx]` records the counter of the most recent _insert for idx
        # (0 == not queued). A popped entry is stale iff its stored counter no
        # longer matches `_latest[idx]` (a later _insert superseded it, or
        # _update_vertex cleared it to 0). This is what lets already-keyed entries
        # survive move_start: their stored keys stay in the heap and the k_m/start
        # drift is compensated inside the pop loop's re-key branch, NOT by
        # deletion. The counter starts at 1 so that 0 unambiguously means
        # "not queued".
        self._heap: list[tuple[float, float, int, int]] = []
        self._latest: list[int] = [0] * self._n
        self._counter: int = 0

        # Goal is the search root: rhs(goal) = 0, everything else +inf.
        self._rhs[self._s_goal_idx] = 0.0
        key = self._calc_key_idx(self._s_goal_idx)
        self._insert(self._s_goal_idx, key)

    # ------------------------------------------------------------------ #
    # Padded-index helpers                                               #
    # ------------------------------------------------------------------ #

    def _to_padded(self, row: int, col: int) -> int:
        """Flat padded index of original-grid ``(row, col)``."""
        return (row + 1) * self._width + (col + 1)

    def _from_padded(self, idx: int) -> Cell:
        """Original-grid ``(row, col)`` of a padded flat index."""
        prow, pcol = divmod(idx, self._width)
        return (prow - 1, pcol - 1)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def calc_key(self, s: Cell) -> Key:
        """The D* Lite priority key for cell ``s`` (public ``(row, col)`` form).

        ``(min(g, rhs) + h(s_start, s) + k_m, min(g, rhs))`` — the lexicographic
        key that orders the open queue.
        """
        return self._calc_key_idx(self._to_padded(int(s[0]), int(s[1])))

    def _calc_key_idx(self, idx: int) -> Key:
        """The priority key for a padded flat index (hot-path form).

        Padding cancels in the coordinate differences, so the octile heuristic
        between two padded indices equals the heuristic between their unpadded
        ``(row, col)`` cells.
        """
        g = self._g[idx]
        rhs = self._rhs[idx]
        g_rhs_min = g if g < rhs else rhs
        return (g_rhs_min + self._h_idx(self._s_start_idx, idx) + self._k_m, g_rhs_min)

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
        g = self._g
        rhs = self._rhs
        edge_meta = self._edge_meta
        interior = self._interior

        while True:
            top = self._peek_valid()
            if top is None:
                # Queue exhausted: no inconsistent vertex remains to relax.
                break

            # `k_old` is the entry's STORED key (its key at insertion time), NOT
            # a recomputed calc_key — that is the whole point of the lazy re-key.
            k0_old, k1_old, u = top
            k_old = (k0_old, k1_old)
            k_new = self._calc_key_idx(u)

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

            start_key = self._calc_key_idx(self._s_start_idx)

            # Figure 3 loop guard: stop once the (now non-stale) top key no longer
            # beats the start key AND the start is locally consistent (rhs == g).
            if not self._key_lt(k_old, start_key) and self._floats_equal(
                rhs[self._s_start_idx], g[self._s_start_idx]
            ):
                break

            self._pop_top()

            g_u = g[u]
            rhs_u = rhs[u]

            if g_u > rhs_u and not self._floats_equal(g_u, rhs_u):
                # Overconsistent: settle g(u) := rhs(u), then UpdateVertex each
                # predecessor (Succ == Pred on this undirected grid).
                g[u] = rhs_u
                for offset, _step, _diag, _oa, _ob in edge_meta:
                    pred = u + offset
                    if interior[pred]:
                        self._update_vertex(pred)
            else:
                # Underconsistent: raise g(u) := inf, then UpdateVertex u AND each
                # predecessor so their rhs re-propagates around the raised vertex.
                g[u] = INF
                self._update_vertex(u)
                for offset, _step, _diag, _oa, _ob in edge_meta:
                    pred = u + offset
                    if interior[pred]:
                        self._update_vertex(pred)

    def update_cells(self, changed_cells: list[Cell]) -> None:
        """Report occupancy flips so incident edge costs and the mirror are repaired.

        For each changed cell, every edge touching it changed cost, so we re-sync
        the occupancy mirror at that cell and then re-evaluate the changed cell
        itself and each of its 8 neighbours via ``update_vertex``. The caller must
        already have mutated ``grid_cells`` (and, when the robot moved, called
        :meth:`move_start`) BEFORE calling this; here we re-read the live value of
        each reported cell into the mirror and fix the affected vertices, then the
        next :meth:`compute_shortest_path` propagates the change.
        """
        if not isinstance(changed_cells, list):
            raise ValueError("changed_cells must be a list of (row, col) tuples.")

        # Collect the changed cells plus their neighbours as padded flat indices,
        # de-duplicated but processed in a deterministic order (sorted). A
        # flat-index sort matches the old (row, col) lexicographic sort, so the
        # deterministic processing order is preserved.
        affected: set[int] = set()
        edge_meta = self._edge_meta
        interior = self._interior
        for cell in changed_cells:
            row = int(cell[0])
            col = int(cell[1])
            if not (0 <= row < self._rows and 0 <= col < self._cols):
                raise ValueError(
                    f"changed cell {cell!r} is outside the grid bounds "
                    f"(rows={self._rows}, cols={self._cols})."
                )
            idx = self._to_padded(row, col)
            # Re-sync the mirror from the live ndarray at this reported flip. This
            # is the ONLY place the mirror is updated, so the report-every-flip
            # contract is load-bearing for occupancy correctness.
            self._occ[idx] = bool(self._grid[row, col])
            affected.add(idx)
            for offset, _step, _diag, _oa, _ob in edge_meta:
                neighbor = idx + offset
                if interior[neighbor]:
                    affected.add(neighbor)

        for idx in sorted(affected):
            self._update_vertex(idx)

    def move_start(self, new_start_cell: Cell) -> None:
        """Slide the search start to ``new_start_cell``.

        Accumulates the heuristic drift into ``k_m`` (so previously computed keys
        stay comparable without re-keying the whole queue) and records the move.
        """
        row = int(new_start_cell[0])
        col = int(new_start_cell[1])
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            raise ValueError(
                f"new_start_cell {new_start_cell!r} is outside the grid bounds "
                f"(rows={self._rows}, cols={self._cols})."
            )

        new_idx = self._to_padded(row, col)
        self._k_m += self._h_idx(self._s_last_idx, new_idx)
        self._s_last_idx = new_idx
        self._s_start_idx = new_idx

    def extract_path(self) -> list[Cell]:
        """Greedily follow the gradient of ``cost(s, s') + g(s')`` to the goal.

        Returns ``[s_start, ..., s_goal]`` as original-grid ``(row, col)`` int
        tuples. Raises ``RuntimeError`` if the start is unreachable
        (``g(s_start)`` is ``inf``) or if a cycle is detected (the iteration cap of
        ``rows * cols`` is exceeded — should never trigger once the search is
        consistent, but guards against a malformed state).
        """
        if not math.isfinite(self._g[self._s_start_idx]):
            raise RuntimeError(
                "No path from start to goal: g(s_start) is infinite."
            )

        g = self._g
        occ = self._occ
        edge_meta = self._edge_meta
        goal_idx = self._s_goal_idx

        path_idx: list[int] = [self._s_start_idx]
        current = self._s_start_idx
        max_iterations = self._rows * self._cols

        for _ in range(max_iterations):
            if current == goal_idx:
                return [self._from_padded(idx) for idx in path_idx]

            best_successor = -1
            best_cost = INF
            for offset, step_cost, is_diagonal, ortho_a, ortho_b in edge_meta:
                successor = current + offset
                # Border successors are occupied in the mirror, so this rejects
                # out-of-bounds moves with no explicit bounds check.
                if occ[successor]:
                    continue
                if is_diagonal and (occ[current + ortho_a] or occ[current + ortho_b]):
                    continue  # no corner cutting
                total = step_cost + g[successor]
                # Strict `<` with deterministic neighbour order means the first
                # minimiser in NEIGHBOR_DELTAS order wins ties — reproducible.
                if total < best_cost:
                    best_cost = total
                    best_successor = successor

            if best_successor < 0 or not math.isfinite(best_cost):
                raise RuntimeError(
                    "No path from start to goal: gradient descent reached a "
                    "dead end with no finite successor."
                )

            current = best_successor
            path_idx.append(current)

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

    def _h_idx(self, a_idx: int, b_idx: int) -> float:
        """Octile distance between two padded flat indices.

        Consistent with the orthogonal=1 / diagonal=sqrt(2) cost model:
        ``max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)``. Padding cancels in the
        coordinate differences, so working on padded indices is exact.
        """
        width = self._width
        ar, ac = divmod(a_idx, width)
        br, bc = divmod(b_idx, width)
        dx = ar - br
        if dx < 0:
            dx = -dx
        dy = ac - bc
        if dy < 0:
            dy = -dy
        if dx >= dy:
            return dx + _OCTILE_DIAGONAL_PENALTY * dy
        return dy + _OCTILE_DIAGONAL_PENALTY * dx

    def _update_vertex(self, u: int) -> None:
        """Recompute ``rhs(u)`` from successors and re-sync ``u``'s queue slot.

        For every cell except the goal, ``rhs(u) = min over successors s of
        cost(u, s) + g(s)`` (inlined here over the 8 padded neighbours, with the
        same occupancy / no-corner-cut rules as :meth:`extract_path`). The vertex
        is then removed from the queue and, if it is locally inconsistent
        (``g != rhs``), re-inserted with a fresh key.
        """
        if u != self._s_goal_idx:
            occ = self._occ
            g = self._g
            best = INF
            for offset, step_cost, is_diagonal, ortho_a, ortho_b in self._edge_meta:
                successor = u + offset
                if occ[successor]:
                    continue  # occupied or border => edge cost inf
                if is_diagonal and (occ[u + ortho_a] or occ[u + ortho_b]):
                    continue  # no corner cutting
                candidate = step_cost + g[successor]
                if candidate < best:
                    best = candidate
            self._rhs[u] = best

        # Remove any existing queue slot (lazy: clear the latest-counter marker;
        # the stale heap entry is skipped on pop because its stored counter no
        # longer matches `_latest[u]` — or, after a fresh _insert, because that
        # later insert superseded it).
        self._latest[u] = 0

        # Re-queue iff locally inconsistent. The tolerant equality keeps a vertex
        # whose g and rhs differ only by a rounding ULP OUT of the queue, so it is
        # never popped and mis-classified as under/overconsistent in the main
        # loop (which would corrupt g with an inf or a stale rhs).
        if not self._floats_equal(self._g[u], self._rhs[u]):
            self._insert(u, self._calc_key_idx(u))

    def _insert(self, idx: int, key: Key) -> None:
        """Push ``idx`` with ``key``, and record this insertion as the cell's
        latest (so any earlier heap entry for it becomes stale)."""
        self._counter += 1
        counter = self._counter
        heapq.heappush(self._heap, (key[0], key[1], counter, idx))
        self._latest[idx] = counter

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
        if math.isinf(left) or math.isinf(right):
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

    def _peek_valid(self) -> tuple[float, float, int] | None:
        """Return ``(stored_key0, stored_key1, cell_idx)`` of the smallest *valid*
        heap entry, or None if the queue is empty.

        Staleness is decided by INSERTION IDENTITY, never by recomputing
        calc_key: an entry is stale iff its stored counter no longer matches the
        cell's latest insertion counter (a newer _insert superseded it, or
        _update_vertex cleared `_latest[idx]` to 0). Valid entries keep their
        stored keys so the k_m/start drift introduced by move_start is compensated
        in the pop loop's re-key branch rather than by dropping still-needed
        vertices.
        """
        heap = self._heap
        latest = self._latest
        while heap:
            k0, k1, stored_counter, idx = heap[0]
            if latest[idx] != stored_counter:
                heapq.heappop(heap)
                continue
            return k0, k1, idx
        return None

    def _pop_top(self) -> None:
        """Pop the validated top entry and clear its queue membership."""
        _, _, _, idx = heapq.heappop(self._heap)
        self._latest[idx] = 0


# --------------------------------------------------------------------------- #
# Controller wiring                                                           #
# --------------------------------------------------------------------------- #


class DStarLiteController:
    """Incremental D* Lite `Controller` over a lidar-folded occupancy grid.

    Plans once at `reset()` from the t=0 lidar fold. On every `act()` it does the
    cheap edge-cost BOOKKEEPING every tick — re-fold the live scan, diff it
    against `self._cells`, mutate `self._cells` in place at the flipped positions,
    `move_start(current_cell)`, and (when cells flipped) `update_cells(changed)` —
    but it DEFERS the expensive tree settle. `compute_shortest_path()` runs only
    at the moment a fresh path is actually needed: when the waypoint follower has
    finished OR the immediate segment it is about to traverse (the robot pose ->
    its current target waypoint) is no longer clear in the live folded grid.

    Why defer the settle. The repaired `g`/`rhs` tree is only ever *consumed* at
    re-extraction, which is rare on a clear run; settling it on every tick was ~89%
    of `act()`'s wallclock (with ~20 moving obstacles each fold flips hundreds of
    inflation-band cells, so the per-tick settle did a large repair whose result
    was thrown away unread), and that cost blew the 600 s per-episode wallclock
    wall on 9 of 50 batch episodes. Deferring the settle is exactly what D* Lite's
    `k_m` machinery exists to support: `move_start` accumulates the heuristic drift
    into `k_m` so stored keys stay comparable across many `update_cells` batches,
    and a single settle at demand-time folds all of those batched edge changes into
    the same optimum a from-scratch A* would find (proved by TC46).

    Committing to a path until its immediate segment is actually blocked also
    avoids a heading whipsaw: re-extracting on every changed cell instead repaints
    a thick inflation band around every wall return, so the optimal cell path
    jitters one or two cells each tick even on a static map, and rebuilding the
    follower from a jittering path (its index reset to the current pose) starves
    forward speed and times the robot out. The commitment horizon keeps the robot
    at full speed on a clear run yet still replans the instant a dynamic obstacle
    crosses in front of it.

    Grid ownership is the load-bearing invariant: `self._cells` is the SAME
    ndarray the search was constructed with. `act()` mutates that array in place
    at the flipped positions and reports them through `update_cells` (which both
    re-syncs the search's occupancy mirror and repairs the affected vertices); it
    never rebinds `self._cells` to a freshly folded array (that would detach the
    search's view and silently desynchronise the occupancy mirror and the
    incremental edge costs). Reporting every flip through `update_cells` is what
    keeps the mirror correct, not merely an incremental-invariant nicety.

    A t=0 planning failure in `reset()` propagates so the runner records
    `planner_error`. A mid-episode failure in `act()` (a deferred settle or
    re-extraction that raises) is swallowed — the last valid follower is kept,
    never rebuilt — so `act()` never raises (AC8).
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
        # The one t=0 settle. Raises RuntimeError if the start cannot reach the
        # goal; the runner turns that into planner_error.
        self._search.compute_shortest_path()
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

        # --- Per-tick edge-cost bookkeeping (cheap; runs every tick) ---------
        new_cells = lidar_to_occupancy(
            self._static_cells, self._grid, state, lidar, self._geom, self._inflation
        )
        diff_mask = self._cells != new_cells

        current_cell = world_to_grid(position, self._grid)
        # move_start is O(1) and bumps k_m by h(s_last, s_new) (== 0 when the robot
        # has not changed cells). Call it UNCONDITIONALLY so the start the deferred
        # settle / extraction reasons about is never stale on a no-change tick.
        # Order within a changed tick stays: move_start FIRST, then update_cells.
        self._search.move_start(current_cell)

        if bool(diff_mask.any()):
            # Deterministic, de-duplicated list of the flipped (row, col) cells.
            changed = sorted(
                (int(row), int(col)) for row, col in zip(*np.where(diff_mask))
            )

            # CRITICAL: mutate the EXISTING array in place — the search holds a
            # reference to it. Do NOT rebind self._cells to new_cells, which would
            # detach the search's view and desynchronise its occupancy mirror and
            # incremental costs. update_cells re-syncs the mirror at these flips.
            self._cells[diff_mask] = new_cells[diff_mask]
            self._search.update_cells(changed)

        # --- Deferred settle, on demand --------------------------------------
        # Re-extract the followed path only when the held one is exhausted or its
        # imminent segment is now blocked — otherwise keep committing to it so a
        # clear run holds full speed instead of chasing the lidar-fold jitter. The
        # expensive compute_shortest_path() runs HERE, not per tick: this is the
        # one place the repaired tree is actually consumed. A failure (settle or
        # extraction) keeps the last valid follower; never rebuild it (AC8).
        if self._follower.is_finished or self._immediate_segment_blocked(position):
            try:
                self._search.compute_shortest_path()
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
                # A failed incremental pass leaves the previous g/rhs and follower
                # in place, so the held path stays drivable.
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
