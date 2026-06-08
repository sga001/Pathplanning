"""DWA controller: a reactive Dynamic Window Approach planner (registry key ``dwa``).

This is a TRUE Dynamic Window Approach (Fox, Burgard & Thrun 1997), not a
single-step lookahead. Every ``act()`` it:

1. builds the *dynamic window* of feasible ``(v, w)`` commands around the last
   command, bounded by the velocity limits and by the acceleration limits over
   one control step;
2. samples a grid of candidate ``(v, w)`` inside that window;
3. FORWARD-SIMULATES the differential-drive motion of each candidate over a
   short multi-step rollout horizon to get a predicted trajectory of poses;
4. scores each trajectory by a weighted sum of heading-to-goal alignment,
   clearance from the live-lidar obstacle cloud, and forward speed, rejecting
   any rollout that comes within the robot radius of an obstacle; and
5. returns the highest-scoring feasible command, clamped to the speed limits.

It carries no global plan: ``reset()`` only caches the goal and the lidar beam
geometry and therefore never raises a ``planner_error``. If every candidate
collides, it falls back to a pure in-place rotation toward the side with more
lidar clearance — no goal-biased escape heuristic (DWA stays a pure Mission
algorithm).

All tunables are the module-level ``UPPER_SNAKE_CASE`` constants below.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manual_astar import (
    MAX_ANGULAR_SPEED,
    MAX_LINEAR_SPEED,
    load_world,
    wrap_to_pi,
)
from planners._grid import LidarGeometry, load_lidar_geometry, register

# --- Tunable constants ------------------------------------------------------

# Control step (s): the dt over which acceleration limits bound the window and
# the dt of each forward-simulation rollout sub-step.
CONTROL_DT = 0.1
# Forward-simulation rollout: number of CONTROL_DT sub-steps simulated per
# candidate (so the predicted trajectory spans ROLLOUT_STEPS * CONTROL_DT s).
ROLLOUT_STEPS = 12

# Dynamic-window acceleration limits (per second). The window is the current
# command +/- (accel limit * CONTROL_DT), intersected with the speed limits.
MAX_LINEAR_ACCEL = 2.0
MAX_ANGULAR_ACCEL = 4.0

# Candidate sampling resolution inside the dynamic window (>= 2 each so the
# window endpoints are always sampled).
LINEAR_SAMPLES = 7
ANGULAR_SAMPLES = 15

# Reverse motion is disallowed (the robot only ever drives forward); the linear
# window is clamped to [MIN_LINEAR_SPEED, MAX_LINEAR_SPEED].
MIN_LINEAR_SPEED = 0.0

# Score weights for the weighted sum (heading + clearance + velocity).
HEADING_WEIGHT = 0.8
CLEARANCE_WEIGHT = 0.3
VELOCITY_WEIGHT = 0.2

# Clearance term: distances beyond CLEARANCE_CAP m saturate (an open trajectory
# should not out-score a goal-aligned one purely on clearance).
CLEARANCE_CAP = 2.0
# Extra safety band added to the robot radius; a rollout whose nearest obstacle
# distance is below (robot_radius + COLLISION_MARGIN) is rejected as colliding.
COLLISION_MARGIN = 0.05
# Within this distance of the goal the heading term is suppressed (the robot is
# essentially on the goal, so heading alignment becomes ill-defined/noisy).
GOAL_REACHED_RADIUS = 0.3
# Below this squared rollout-segment length the final heading is treated as
# undefined (a near-stationary candidate has no meaningful direction).
MIN_ROLLOUT_STEP_SQ = 1e-12

# In-place-rotation fallback when every candidate collides. The turn direction
# is chosen toward the side (left vs right half of the scan) with more mean
# clearance; magnitude is a fixed fraction of the angular limit.
FALLBACK_TURN_RATE = 0.6


@dataclass(frozen=True)
class _Window:
    """The feasible ``(v, w)`` dynamic window for one control step.

    Each bound is the current command +/- (accel limit * CONTROL_DT), already
    intersected with the absolute speed limits, so sampling within these bounds
    needs no further clamping.
    """

    v_min: float
    v_max: float
    w_min: float
    w_max: float


class DWAController:
    """Reactive Dynamic Window Approach controller (registry key ``dwa``).

    Holds the last commanded ``(v, w)`` as state so the dynamic window can be
    built around it each tick (initialized to ``(0, 0)``). Reactive: no global
    plan, so ``reset()`` never raises.
    """

    name = "dwa"

    def __init__(self, replan_k: int | None = None) -> None:
        # `build_controller` rejects a non-None `replan_k` for non-replan
        # families before construction; the kwarg is accepted here only to match
        # the uniform `ALGORITHMS[name](replan_k=...)` construction seam, then
        # ignored (mirrors AStarOnceController.__init__).
        del replan_k

        # Static substrate, populated by reset().
        self._goal_xy: np.ndarray | None = None
        self._geom: LidarGeometry | None = None
        self._bearings: np.ndarray | None = None
        self._robot_radius: float | None = None

        # Last commanded velocities (the dynamic window is centered here).
        self._last_v = 0.0
        self._last_w = 0.0

    def reset(
        self,
        world_yaml: str,
        initial_snapshot: tuple,
        lidar0: np.ndarray,
        state0: np.ndarray,
    ) -> None:
        """Cache the goal xy and the lidar beam bearings. Never raises.

        Reactive planners have no global plan, so there is nothing here that can
        fail in a way the runner should record as a planner_error; the live
        snapshot, the t=0 lidar frame, and the start pose carry no information
        DWA uses up front.
        """
        del initial_snapshot, lidar0, state0

        world = load_world(world_yaml)
        # `goal` is [x, y]; only the xy is used for goal-seeking.
        self._goal_xy = np.asarray(world.goal, dtype=float)[:2]
        # The lidar sits at the robot CENTER and returns center-to-surface
        # distances, so the body radius must be subtracted to recover true
        # clearance and added to the collision-rejection band (see C1).
        self._robot_radius = float(world.robot_radius)

        self._geom = load_lidar_geometry(world_yaml)
        # irsim lays beams with linspace over the inclusive [angle_min, angle_max]
        # endpoints — the same recovery `lidar_to_occupancy` uses.
        self._bearings = np.linspace(
            self._geom.angle_min, self._geom.angle_max, self._geom.number
        )

        # A fresh episode starts from rest.
        self._last_v = 0.0
        self._last_w = 0.0

    def act(self, state: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        """Pick the best feasible ``(v, w)`` over the dynamic window.

        Returns a ``(2, 1)`` float64 action ``[[v], [w]]`` clamped to the speed
        limits. Never raises on a per-tick failure: if no candidate is feasible
        it falls back to an in-place rotation toward the clearer side.
        """
        if (
            self._goal_xy is None
            or self._geom is None
            or self._bearings is None
            or self._robot_radius is None
        ):
            raise RuntimeError("act() called before reset().")
        if state.shape != (3,):
            raise ValueError(
                f"Expected (3,) [x, y, theta] state, received shape {state.shape}."
            )
        if lidar.shape != (self._geom.number,):
            raise ValueError(
                f"Expected lidar of shape {(self._geom.number,)}, received {lidar.shape}."
            )

        obstacle_points = self._lidar_to_world_points(state, lidar)
        window = self._dynamic_window()

        best_v: float | None = None
        best_w = 0.0
        best_score = -np.inf

        for candidate_v in np.linspace(window.v_min, window.v_max, LINEAR_SAMPLES):
            for candidate_w in np.linspace(window.w_min, window.w_max, ANGULAR_SAMPLES):
                trajectory = self._rollout(state, float(candidate_v), float(candidate_w))
                clearance = self._trajectory_clearance(trajectory, obstacle_points)
                if clearance is None:
                    # Rollout collides (within robot radius + margin): reject.
                    continue

                score = self._score(trajectory, float(candidate_v), clearance)
                if score > best_score:
                    best_score = score
                    best_v = float(candidate_v)
                    best_w = float(candidate_w)

        if best_v is None:
            # Every candidate collides: rotate in place toward the clearer side.
            return self._fallback_action(lidar)

        # Defensive clamp (the window is already within limits, but float drift
        # at the endpoints must never push the command past the hard limits).
        clamped_v = float(np.clip(best_v, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED))
        clamped_w = float(np.clip(best_w, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))
        self._last_v = clamped_v
        self._last_w = clamped_w
        return np.array([[clamped_v], [clamped_w]], dtype=float)

    # --- Internal helpers ---------------------------------------------------

    def _dynamic_window(self) -> _Window:
        """The feasible ``(v, w)`` window around the last command for one step.

        Each axis is the last command +/- (accel limit * CONTROL_DT), clamped to
        the absolute speed limits (and the linear axis to a non-negative floor so
        the robot never commands reverse).
        """
        dv = MAX_LINEAR_ACCEL * CONTROL_DT
        dw = MAX_ANGULAR_ACCEL * CONTROL_DT

        v_min = max(MIN_LINEAR_SPEED, self._last_v - dv)
        v_max = min(MAX_LINEAR_SPEED, self._last_v + dv)
        w_min = max(-MAX_ANGULAR_SPEED, self._last_w - dw)
        w_max = min(MAX_ANGULAR_SPEED, self._last_w + dw)

        # Guard against an empty/inverted window from float drift at the limits.
        if v_max < v_min:
            v_max = v_min
        if w_max < w_min:
            w_max = w_min

        return _Window(v_min=v_min, v_max=v_max, w_min=w_min, w_max=w_max)

    def _rollout(self, state: np.ndarray, v: float, w: float) -> np.ndarray:
        """Forward-simulate constant ``(v, w)`` over the rollout horizon.

        Returns an ``(ROLLOUT_STEPS, 2)`` array of predicted xy positions
        (excluding the start pose) under the exact differential-drive unicycle
        update at CONTROL_DT per sub-step.
        """
        x = float(state[0])
        y = float(state[1])
        theta = float(state[2])

        positions = np.empty((ROLLOUT_STEPS, 2), dtype=float)
        for step_index in range(ROLLOUT_STEPS):
            theta = wrap_to_pi(theta + w * CONTROL_DT)
            x += v * np.cos(theta) * CONTROL_DT
            y += v * np.sin(theta) * CONTROL_DT
            positions[step_index, 0] = x
            positions[step_index, 1] = y

        return positions

    def _trajectory_clearance(
        self, trajectory: np.ndarray, obstacle_points: np.ndarray
    ) -> float | None:
        """True body clearance of a rollout against the lidar obstacle cloud.

        In this harness the lidar sits at the robot CENTER and the arena passes
        ``ranges`` through raw, so each return is a CENTER-to-SURFACE distance;
        the rollout poses are also center positions. The nearest center-to-
        surface distance therefore overstates clearance by the body radius, so
        we subtract ``robot_radius`` to get the true gap between the robot body
        and the obstacle surface. A candidate is rejected (returns ``None``) when
        that body clearance drops to within COLLISION_MARGIN — i.e. the 0.2 m
        body would overlap or graze the obstacle. With no obstacle points the
        clearance saturates at CLEARANCE_CAP.
        """
        assert self._robot_radius is not None  # narrowed by act()'s guard

        if obstacle_points.shape[0] == 0:
            return CLEARANCE_CAP

        # Pairwise min distance: (ROLLOUT_STEPS, 1, 2) - (1, N, 2) -> (R, N).
        deltas = trajectory[:, None, :] - obstacle_points[None, :, :]
        distances = np.sqrt(np.einsum("rnk,rnk->rn", deltas, deltas))
        min_distance = float(distances.min())

        # Reject when the body (center +/- radius) comes within the safety band.
        if min_distance <= self._robot_radius + COLLISION_MARGIN:
            return None
        # The score's clearance term reflects true body clearance, not the
        # center-to-surface distance, before the cap/normalize.
        body_clearance = min_distance - self._robot_radius
        return min(body_clearance, CLEARANCE_CAP)

    def _score(self, trajectory: np.ndarray, v: float, clearance: float) -> float:
        """Weighted sum of heading-to-goal, normalized clearance, and speed.

        - Heading: 1 when the rollout's final pose points straight at the goal,
          0 when it points directly away (``(pi - |error|) / pi``). Suppressed
          when the final pose is essentially on the goal (heading is then noise).
        - Clearance: ``clearance / CLEARANCE_CAP`` in ``[0, 1]``.
        - Velocity: ``v / MAX_LINEAR_SPEED`` in ``[0, 1]``.
        """
        assert self._goal_xy is not None  # narrowed by act()'s guard

        final_position = trajectory[-1]
        to_goal = self._goal_xy - final_position
        goal_distance = float(np.linalg.norm(to_goal))

        if goal_distance < GOAL_REACHED_RADIUS:
            heading_term = 1.0
        else:
            # Estimate the rollout's final heading from its last segment so the
            # alignment reflects where the trajectory is actually pointing.
            if trajectory.shape[0] >= 2:
                step = trajectory[-1] - trajectory[-2]
            else:
                step = trajectory[-1] - final_position
            if float(np.dot(step, step)) < MIN_ROLLOUT_STEP_SQ:
                # No motion (v ~ 0): use the bearing-to-goal angular error of 0
                # so a stationary rollout neither gains nor loses on heading.
                heading_error = np.pi
            else:
                final_heading = np.arctan2(step[1], step[0])
                goal_bearing = np.arctan2(to_goal[1], to_goal[0])
                heading_error = abs(wrap_to_pi(goal_bearing - final_heading))
            heading_term = (np.pi - heading_error) / np.pi

        clearance_term = clearance / CLEARANCE_CAP
        velocity_term = v / MAX_LINEAR_SPEED

        return (
            HEADING_WEIGHT * heading_term
            + CLEARANCE_WEIGHT * clearance_term
            + VELOCITY_WEIGHT * velocity_term
        )

    def _lidar_to_world_points(
        self, state: np.ndarray, lidar: np.ndarray
    ) -> np.ndarray:
        """Project finite lidar returns to world-frame obstacle points.

        For beam ``i`` with finite range ``r``: world bearing is
        ``theta + bearings[i]`` and the hit is ``(x + r*cos, y + r*sin)``. NaN
        (no-return) beams are skipped. Returns an ``(N, 2)`` array (possibly
        empty).
        """
        assert self._bearings is not None  # narrowed by act()'s guard

        ranges = np.asarray(lidar, dtype=float)
        finite_mask = np.isfinite(ranges)
        if not finite_mask.any():
            return np.empty((0, 2), dtype=float)

        finite_ranges = ranges[finite_mask]
        world_angles = float(state[2]) + self._bearings[finite_mask]
        hit_x = float(state[0]) + finite_ranges * np.cos(world_angles)
        hit_y = float(state[1]) + finite_ranges * np.sin(world_angles)
        return np.column_stack((hit_x, hit_y))

    def _fallback_action(self, lidar: np.ndarray) -> np.ndarray:
        """In-place rotation toward the side with more mean lidar clearance.

        Pure DWA escape: ``v = 0`` and ``w`` toward whichever half of the scan
        has greater mean finite range (NaN treated as max clearance). No
        goal-biased escape heuristic. Updates the stored command so the next
        dynamic window is centered on this rotation.
        """
        ranges = np.asarray(lidar, dtype=float)
        # NaN = no return = maximally clear; treat it as the clearance cap so an
        # empty side is preferred over a side full of close returns.
        filled = np.where(np.isfinite(ranges), ranges, CLEARANCE_CAP)

        half = filled.shape[0] // 2
        right_mean = float(filled[:half].mean()) if half > 0 else 0.0
        left_mean = float(filled[half:].mean()) if filled.shape[0] - half > 0 else 0.0

        # Beams are ordered from angle_min (right, negative) to angle_max (left,
        # positive); positive w turns toward +theta (left).
        turn_rate = FALLBACK_TURN_RATE * MAX_ANGULAR_SPEED
        angular = turn_rate if left_mean >= right_mean else -turn_rate
        angular = float(np.clip(angular, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))

        self._last_v = 0.0
        self._last_w = angular
        return np.array([[0.0], [angular]], dtype=float)


register("dwa", DWAController)
