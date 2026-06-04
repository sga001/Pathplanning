"""Phase 2 crossing-traffic substrate for the Arena harness.

This module provides `DynamicObstacleState` (a frozen per-tick snapshot record)
and `TrafficSpawner` (the live spawner/advancer/despawner/refiller). It is
designed to be deterministic under a fixed `traffic_rng` seed: same seed +
same static world + same step count must produce the same sequence of
`state_sha256()` digests.

Determinism verification (T6, completed 2026-05-28): TC20 in
`arena/arena.py --check` runs two `Arena(seed=3, traffic=True)` instances over
200 zero-action ticks and asserts byte-identical `dynamic_obstacles_sha256`
sequences (including the reset-time hash). Result on the dev laptop: PASS.
The stronger per-tick `lidar_sha256` byte-identity sub-goal (AC4 stretch)
was NOT separately asserted — only the spawner-side hash is pinned. If
TC15-style trace JSONL determinism is verified under `--traffic` in a
future task, this comment should record that result too.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from arena.arena import ArenaRuntimeError


TARGET_POPULATION = 20
OBSTACLE_RADIUS = 0.3
SPEED_MIN_FACTOR = 0.3
SPEED_MAX_FACTOR = 1.5
SPAWN_OVERLAP_BUFFER = 1.0
DESPAWN_BUFFER = 0.5
# Generous attempt budget: a spawn almost always succeeds in 1-2 tries, so a high
# cap makes a silent short-population (the TC18 invariant) effectively impossible
# without changing RNG consumption on the common path.
SPAWN_MAX_ATTEMPTS = 100
DYNAMIC_OBSTACLE_NAME_FMT = "traffic_{idx}"


@dataclass(frozen=True)
class DynamicObstacleState:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float


@dataclass
class _LiveObstacle:
    """Internal bookkeeping record. Velocity AND position are owned by the
    spawner: irsim omni objects do not expose vx/vy directly, and we
    deliberately do NOT read x/y back from the irsim handle either — keeping
    a spawner-side cache rules out determinism coupling to irsim's float
    round-tripping through its state-storage path (AC4)."""

    handle: Any
    x: float
    y: float
    vx: float
    vy: float
    radius: float


class TrafficSpawner:
    """Spawns, advances, despawns, and refills circular dynamic obstacles
    around a square arena perimeter. Deterministic under fixed RNG state.

    `live_ids` and `_next_idx` are intentionally distinct: `live_ids`
    reflects which irsim object ids exist *right now*; `_next_idx` is a
    monotonically increasing counter so obstacle names never collide across
    an Arena's lifetime (even after delete + respawn).
    """

    def __init__(
        self,
        env: Any,
        robot: Any,
        traffic_rng: np.random.Generator,
        motion_rng: np.random.Generator,
        dt: float,
        arena_w: float,
        arena_h: float,
        static_obstacles: Sequence[Any],
    ) -> None:
        self._env = env
        self._robot = robot
        self._traffic_rng = traffic_rng
        self._motion_rng = motion_rng  # plumbed for forward-compat; unused in Phase 2
        self._dt = float(dt)
        self._arena_w = float(arena_w)
        self._arena_h = float(arena_h)
        self._static_obstacles = list(static_obstacles)

        robot_state = self._robot.state
        self._robot_start_xy = np.array(
            [float(robot_state[0, 0]), float(robot_state[1, 0])], dtype=np.float64
        )

        self._live: dict[int, _LiveObstacle] = {}
        self._next_idx = 0
        self._closed = False

        # Cache the point-to-obstacle distance callable once per spawner lifetime.
        # Lazy import to avoid a hard dependency cycle: arena.arena -> arena.dynamic
        # at module import time. Mirrors the TC10 sys.path pattern in arena/arena.py.
        import sys as _sys
        from pathlib import Path as _Path

        _repo_root = str(_Path(__file__).resolve().parent.parent)
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from manual_astar import (  # type: ignore[import-not-found]
            MAX_LINEAR_SPEED,
            point_to_obstacle_distance,
        )

        self._point_to_obstacle_distance = point_to_obstacle_distance
        # Source the obstacle speed band from the robot's actual top speed instead of
        # a hand-copied constant, so the two cannot silently drift apart.
        self._robot_top_speed = float(MAX_LINEAR_SPEED)

    @property
    def live_ids(self) -> frozenset[int]:
        return frozenset(self._live.keys())

    def reset(
        self,
        traffic_rng: np.random.Generator,
        motion_rng: np.random.Generator,
    ) -> None:
        """Clear all live obstacles and rebind the RNGs for a new episode WITHOUT
        rebuilding the spawner: the cached point-distance callable, static-obstacle
        list, and cached robot start stay put (env.reset() restores the same robot
        pose). _next_idx is intentionally NOT reset, so obstacle names never collide
        across the Arena's lifetime even after delete + respawn."""
        if self._closed:
            raise ArenaRuntimeError("TrafficSpawner.reset() called after close()")
        for obs_id in list(self._live.keys()):
            try:
                self._env.delete_object(obs_id)
            except (KeyError, ValueError, AttributeError):
                pass
        self._live = {}
        self._traffic_rng = traffic_rng
        self._motion_rng = motion_rng

    def initialize(self) -> tuple[DynamicObstacleState, ...]:
        self._refill()
        return self.snapshot()

    def step(self) -> tuple[DynamicObstacleState, ...]:
        self._advance()
        self._despawn()
        self._refill()
        return self.snapshot()

    def snapshot(self) -> tuple[DynamicObstacleState, ...]:
        out: list[DynamicObstacleState] = []
        for obs_id in sorted(self._live.keys()):
            live = self._live[obs_id]
            out.append(
                DynamicObstacleState(
                    id=obs_id,
                    x=live.x,
                    y=live.y,
                    vx=live.vx,
                    vy=live.vy,
                    radius=live.radius,
                )
            )
        return tuple(out)

    def state_sha256(
        self, snap: tuple[DynamicObstacleState, ...] | None = None
    ) -> str:
        # Accept an already-built snapshot so callers don't rebuild it (step() and
        # initialize() already produced one). Hash physical state only: obstacle id is
        # an irsim handle that climbs across reset() (id_iter resets per env.make(),
        # not per env.reset()), so hashing it would make the digest differ across
        # repeated reset() of one Arena. Row order stays by id via snapshot(), so the
        # ordering is still deterministic.
        if snap is None:
            snap = self.snapshot()
        if not snap:
            arr = np.empty((0, 5), dtype=np.float64)
        else:
            arr = np.array(
                [[s.x, s.y, s.vx, s.vy, s.radius] for s in snap],
                dtype=np.float64,
            )
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def close(self) -> None:
        if self._closed:
            return
        for obs_id in list(self._live.keys()):
            try:
                self._env.delete_object(obs_id)
            except (KeyError, ValueError, AttributeError):
                # id-not-found is expected during torn-down env; other errors are programmer bugs we want to surface — but close() must not raise per Arena.close() contract.
                pass
        self._live = {}
        self._closed = True

    def _inject_for_test(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        radius: float = OBSTACLE_RADIUS,
    ) -> DynamicObstacleState:
        """Spawn an obstacle at an explicit state without drawing from traffic_rng.

        Mirrors the create_obstacle + add_object + record flow of a normal spawn,
        but does NOT consume any RNG draws so subsequent normal spawns see the
        same RNG state they would have seen without the injection.
        """
        if not all(math.isfinite(v) for v in (x, y, vx, vy, radius)):
            raise ValueError(
                f"_inject_for_test got non-finite values: x={x}, y={y}, vx={vx}, vy={vy}, radius={radius}"
            )
        if radius <= 0:
            raise ValueError(f"_inject_for_test requires radius > 0, got {radius}")
        handle = self._create_and_attach(x, y, radius)
        self._live[handle.id] = _LiveObstacle(
            handle=handle,
            x=float(x),
            y=float(y),
            vx=float(vx),
            vy=float(vy),
            radius=float(radius),
        )
        return DynamicObstacleState(
            id=handle.id, x=float(x), y=float(y), vx=float(vx), vy=float(vy), radius=float(radius)
        )

    def _advance(self) -> None:
        for live in self._live.values():
            # Update the spawner-side cache FIRST (source of truth for determinism),
            # then push to the irsim handle once for lidar/collision consumers.
            live.x = live.x + live.vx * self._dt
            live.y = live.y + live.vy * self._dt
            self._write_xy(live.handle, live.x, live.y)

    def _despawn(self) -> None:
        lo_x = -DESPAWN_BUFFER
        hi_x = self._arena_w + DESPAWN_BUFFER
        lo_y = -DESPAWN_BUFFER
        hi_y = self._arena_h + DESPAWN_BUFFER

        to_remove: list[int] = []
        for obs_id, live in self._live.items():
            if live.x < lo_x or live.x > hi_x or live.y < lo_y or live.y > hi_y:
                to_remove.append(obs_id)

        for obs_id in to_remove:
            # Reconcile our own tracking FIRST so a delete_object failure cannot leave
            # a phantom id in _live (which snapshot()/state_sha256() would then
            # over-report relative to what irsim's lidar/collision tree actually sees).
            del self._live[obs_id]
            try:
                self._env.delete_object(obs_id)
            except Exception as exc:
                raise ArenaRuntimeError(
                    f"env.delete_object failed for tracked id {obs_id}: {exc}"
                ) from exc

    def _refill(self) -> None:
        while len(self._live) < TARGET_POPULATION:
            spawned = self._try_one_spawn()
            if not spawned:
                # Exhausted SPAWN_MAX_ATTEMPTS for this slot. Surface it instead of
                # silently shipping a short population (the harness/TC18 treat 20 as an
                # invariant) so an unlucky seed is debuggable. Non-fatal: next tick retries.
                warnings.warn(
                    f"TrafficSpawner: refill gave up at {len(self._live)}/"
                    f"{TARGET_POPULATION} live after {SPAWN_MAX_ATTEMPTS} attempts; "
                    "population is below target this tick.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return

    def _try_one_spawn(self) -> bool:
        for _ in range(SPAWN_MAX_ATTEMPTS):
            # Three RNG draws per attempt, always in this order: position, heading, speed.
            t = float(
                self._traffic_rng.uniform(0.0, 2.0 * (self._arena_w + self._arena_h))
            )
            x, y, heading_lo, heading_hi = self._perimeter_sample(t)
            heading = float(self._traffic_rng.uniform(heading_lo, heading_hi))
            speed = float(
                self._traffic_rng.uniform(SPEED_MIN_FACTOR, SPEED_MAX_FACTOR)
                * self._robot_top_speed
            )

            if self._overlaps_robot_start(x, y):
                continue
            if self._overlaps_static(x, y):
                continue

            vx = speed * math.cos(heading)
            vy = speed * math.sin(heading)

            handle = self._create_and_attach(x, y, OBSTACLE_RADIUS)
            self._live[handle.id] = _LiveObstacle(
                handle=handle,
                x=float(x),
                y=float(y),
                vx=vx,
                vy=vy,
                radius=OBSTACLE_RADIUS,
            )
            return True
        return False

    def _perimeter_sample(self, t: float) -> tuple[float, float, float, float]:
        """Map t in [0, 2*(W+H)) onto the arena perimeter and return the inward
        heading half-cone for that edge. Each edge is mapped by its own length
        (south=W, east=H, north=W, west=H), so this works for any rectangle, not
        only the square arenas shipped today."""
        W = self._arena_w
        H = self._arena_h
        if t < W:
            return (t, 0.0, 0.0, math.pi)
        if t < W + H:
            return (W, t - W, math.pi / 2.0, 3.0 * math.pi / 2.0)
        if t < 2.0 * W + H:
            return (2.0 * W + H - t, H, math.pi, 2.0 * math.pi)
        return (0.0, 2.0 * W + 2.0 * H - t, -math.pi / 2.0, math.pi / 2.0)

    def _overlaps_robot_start(self, x: float, y: float) -> bool:
        dx = x - self._robot_start_xy[0]
        dy = y - self._robot_start_xy[1]
        return math.hypot(dx, dy) < OBSTACLE_RADIUS + SPAWN_OVERLAP_BUFFER

    def _overlaps_static(self, x: float, y: float) -> bool:
        if not self._static_obstacles:
            return False
        point = np.array([x, y], dtype=np.float64)
        threshold = OBSTACLE_RADIUS + SPAWN_OVERLAP_BUFFER
        for static_obs in self._static_obstacles:
            if self._point_to_obstacle_distance(point, static_obs) < threshold:
                return True
        return False

    def _create_and_attach(self, x: float, y: float, radius: float) -> Any:
        name = DYNAMIC_OBSTACLE_NAME_FMT.format(idx=self._next_idx)
        self._next_idx += 1
        try:
            obs = self._env.create_obstacle(
                kinematics={"name": "omni"},
                shape={"name": "circle", "radius": radius},
                state=[float(x), float(y), 0.0],
                name=name,
            )
            self._env.add_object(obs)
        except ValueError as exc:
            raise ArenaRuntimeError(
                f"env.add_object rejected obstacle name {name!r}: {exc}"
            ) from exc
        return obs

    @staticmethod
    def _write_xy(handle: Any, x: float, y: float) -> None:
        # ObjectBase.state is read-only; the public mutation API is set_state(state, init=False).
        # Passing init=False keeps env.reset()'s spawn-pose restore behavior untouched.
        handle.set_state([float(x), float(y), 0.0], init=False)
