from __future__ import annotations

import argparse
import heapq
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import irsim
import numpy as np
import yaml


GRID_RESOLUTION = 0.1
SAFETY_MARGIN = 0.15
WAYPOINT_STRIDE = 5
WAYPOINT_REACHED_DISTANCE = 0.25
FINAL_GOAL_TOLERANCE = 0.1
MAX_LINEAR_SPEED = 1.0
MIN_LINEAR_SPEED = 0.2
MAX_ANGULAR_SPEED = 1.0
MAX_STEPS = 1000


@dataclass(frozen=True)
class ObstacleSpec:
    kind: str
    center: np.ndarray | None = None
    radius: float = 0.0
    vertices: np.ndarray | None = None


@dataclass(frozen=True)
class WorldModel:
    width: float
    height: float
    offset: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    robot_radius: float
    obstacles: tuple[ObstacleSpec, ...]


@dataclass(frozen=True)
class OccupancyGrid:
    cells: np.ndarray
    resolution: float
    offset: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.cells.shape


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotation_matrix(theta: float) -> np.ndarray:
    cosine = np.cos(theta)
    sine = np.sin(theta)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=float)


def closest_point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    segment_length_sq = float(np.dot(segment, segment))

    if segment_length_sq < 1e-9:
        return start

    projection = float(np.dot(point - start, segment) / segment_length_sq)
    projection = np.clip(projection, 0.0, 1.0)
    return start + projection * segment


def point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    x_coord, y_coord = point
    inside = False
    count = len(vertices)

    for index in range(count):
        x1, y1 = vertices[index]
        x2, y2 = vertices[(index + 1) % count]
        intersects = (y1 > y_coord) != (y2 > y_coord)

        if not intersects:
            continue

        x_intersection = (x2 - x1) * (y_coord - y1) / (y2 - y1 + 1e-12) + x1
        if x_coord < x_intersection:
            inside = not inside

    return inside


def point_to_polyline_distance(point: np.ndarray, vertices: np.ndarray, closed: bool) -> float:
    if len(vertices) == 1:
        return float(np.linalg.norm(point - vertices[0]))

    segment_count = len(vertices) if closed else len(vertices) - 1
    min_distance = np.inf

    for index in range(segment_count):
        start = vertices[index]
        end = vertices[(index + 1) % len(vertices)] if closed else vertices[index + 1]
        closest = closest_point_on_segment(point, start, end)
        min_distance = min(min_distance, float(np.linalg.norm(point - closest)))

    return min_distance


def point_to_obstacle_distance(point: np.ndarray, obstacle: ObstacleSpec) -> float:
    if obstacle.kind == 'circle':
        if obstacle.center is None:
            raise ValueError('Circle obstacles require a center point.')
        return float(np.linalg.norm(point - obstacle.center) - obstacle.radius)

    if obstacle.kind == 'linestring':
        if obstacle.vertices is None:
            raise ValueError('Linestring obstacles require vertices.')
        return point_to_polyline_distance(point, obstacle.vertices, closed=False)

    if obstacle.kind in {'polygon', 'rectangle'}:
        if obstacle.vertices is None:
            raise ValueError('Polygon obstacles require vertices.')
        if point_in_polygon(point, obstacle.vertices):
            return 0.0

        return point_to_polyline_distance(point, obstacle.vertices, closed=True)

    raise ValueError(f'Unsupported obstacle kind: {obstacle.kind}')


def parse_state(raw_state: list[float] | None) -> np.ndarray:
    if raw_state is None:
        return np.zeros(3, dtype=float)

    state = np.asarray(raw_state, dtype=float)
    if state.shape != (3,):
        raise ValueError(f'Expected [x, y, theta] state, received {raw_state!r}.')

    return state


def transform_vertices(raw_vertices: list[list[float]], state: np.ndarray) -> np.ndarray:
    vertices = np.asarray(raw_vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError(f'Expected Nx2 vertices, received {raw_vertices!r}.')

    transformed = vertices @ rotation_matrix(state[2]).T
    transformed += state[:2]
    return transformed


def rectangle_vertices(length: float, width: float, state: np.ndarray) -> np.ndarray:
    half_length = length / 2.0
    half_width = width / 2.0
    local_vertices = np.array(
        [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )
    return transform_vertices(local_vertices.tolist(), state)


def parse_obstacle(raw_obstacle: dict) -> ObstacleSpec:
    shape = raw_obstacle['shape']
    kind = shape['name']
    state = parse_state(raw_obstacle.get('state'))

    if kind == 'circle':
        return ObstacleSpec(kind='circle', center=state[:2], radius=float(shape['radius']))

    if kind == 'rectangle':
        vertices = rectangle_vertices(float(shape['length']), float(shape['width']), state)
        return ObstacleSpec(kind='rectangle', vertices=vertices)

    if kind == 'polygon':
        raw_vertices = shape['vertices']
        vertices = np.asarray(raw_vertices, dtype=float)
        if 'state' in raw_obstacle:
            vertices = transform_vertices(raw_vertices, state)
        return ObstacleSpec(kind='polygon', vertices=vertices)

    if kind == 'linestring':
        vertices = transform_vertices(shape['vertices'], state)
        return ObstacleSpec(kind='linestring', vertices=vertices)

    raise ValueError(f'Unsupported obstacle kind: {kind}')


def load_world(map_path: str) -> WorldModel:
    raw_world = yaml.safe_load(Path(map_path).read_text(encoding='utf-8'))
    world_section = raw_world.get('world', {})
    robot_section = raw_world['robot']
    robot_shape = robot_section['shape']

    offset = np.asarray(world_section.get('offset', [0.0, 0.0]), dtype=float)
    if offset.shape != (2,):
        raise ValueError(f'Expected [x, y] world offset, received {offset!r}.')

    obstacles = tuple(parse_obstacle(raw_obstacle) for raw_obstacle in raw_world.get('obstacle', []))

    return WorldModel(
        width=float(world_section['width']),
        height=float(world_section['height']),
        offset=offset,
        start=np.asarray(robot_section['state'][:2], dtype=float),
        goal=np.asarray(robot_section['goal'][:2], dtype=float),
        robot_radius=float(robot_shape.get('radius', 0.0)),
        obstacles=obstacles,
    )


def world_to_grid(point: np.ndarray, grid: OccupancyGrid) -> tuple[int, int]:
    rows, cols = grid.shape
    raw_x = (point[0] - grid.offset[0]) / grid.resolution
    raw_y = (point[1] - grid.offset[1]) / grid.resolution
    grid_x = int(np.clip(np.floor(raw_x), 0, cols - 1))
    grid_y = int(np.clip(np.floor(raw_y), 0, rows - 1))
    return grid_y, grid_x


def grid_to_world(cell: tuple[int, int], grid: OccupancyGrid) -> np.ndarray:
    row, col = cell
    x_coord = grid.offset[0] + (col + 0.5) * grid.resolution
    y_coord = grid.offset[1] + (row + 0.5) * grid.resolution
    return np.array([x_coord, y_coord], dtype=float)


def is_cell_in_bounds(cell: tuple[int, int], grid: OccupancyGrid) -> bool:
    row, col = cell
    rows, cols = grid.shape
    return 0 <= row < rows and 0 <= col < cols


def validate_world_point(point: np.ndarray, world: WorldModel) -> None:
    min_corner = world.offset
    max_corner = world.offset + np.array([world.width, world.height], dtype=float)

    if np.any(point < min_corner) or np.any(point > max_corner):
        raise ValueError(f'Point {point.tolist()} is outside the world bounds.')


def build_occupancy_grid(world: WorldModel, resolution: float, inflation_margin: float) -> OccupancyGrid:
    columns = int(np.ceil(world.width / resolution))
    rows = int(np.ceil(world.height / resolution))
    cells = np.zeros((rows, columns), dtype=bool)
    inflation_radius = world.robot_radius + inflation_margin
    grid = OccupancyGrid(cells=cells, resolution=resolution, offset=world.offset)

    for row in range(rows):
        for col in range(columns):
            point = grid_to_world((row, col), grid)
            occupied = is_point_blocked(point, world.obstacles, inflation_radius)
            cells[row, col] = occupied

    return grid


def is_point_blocked(point: np.ndarray, obstacles: tuple[ObstacleSpec, ...], inflation_radius: float) -> bool:
    return any(point_to_obstacle_distance(point, obstacle) <= inflation_radius for obstacle in obstacles)


def validate_start_and_goal(world: WorldModel, grid: OccupancyGrid) -> tuple[tuple[int, int], tuple[int, int]]:
    validate_world_point(world.start, world)
    validate_world_point(world.goal, world)

    inflation_radius = world.robot_radius + SAFETY_MARGIN
    if is_point_blocked(world.start, world.obstacles, inflation_radius):
        raise ValueError('The start position is blocked after obstacle inflation.')

    if is_point_blocked(world.goal, world.obstacles, inflation_radius):
        raise ValueError('The goal position is blocked after obstacle inflation.')

    start_cell = world_to_grid(world.start, grid)
    goal_cell = world_to_grid(world.goal, grid)

    if not is_cell_in_bounds(start_cell, grid):
        raise ValueError('The start position is outside the occupancy grid.')

    if not is_cell_in_bounds(goal_cell, grid):
        raise ValueError('The goal position is outside the occupancy grid.')

    if grid.cells[start_cell]:
        raise ValueError('The start position is blocked after obstacle inflation.')

    if grid.cells[goal_cell]:
        raise ValueError('The goal position is blocked after obstacle inflation.')

    return start_cell, goal_cell


def heuristic(cell: tuple[int, int], goal: tuple[int, int]) -> float:
    return float(np.hypot(goal[0] - cell[0], goal[1] - cell[1]))


def reconstruct_path(
    parents: dict[tuple[int, int], tuple[int, int] | None],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [goal]
    current = goal

    while True:
        parent = parents[current]
        if parent is None:
            break
        current = parent
        path.append(current)

    path.reverse()
    return path


def astar_search(
    grid: OccupancyGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    heuristic_fn: Callable[[tuple[int, int], tuple[int, int]], float] | None = None,
) -> list[tuple[int, int]]:
    h = heuristic_fn if heuristic_fn is not None else heuristic

    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (h(start, goal), start))

    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            return reconstruct_path(parents, goal)

        current_cost = g_score[current]

        for delta_row, delta_col in neighbors:
            neighbor = (current[0] + delta_row, current[1] + delta_col)

            if not is_cell_in_bounds(neighbor, grid) or grid.cells[neighbor]:
                continue

            if delta_row != 0 and delta_col != 0:
                row_neighbor = (current[0] + delta_row, current[1])
                col_neighbor = (current[0], current[1] + delta_col)
                if grid.cells[row_neighbor] or grid.cells[col_neighbor]:
                    continue

            step_cost = float(np.hypot(delta_row, delta_col))
            tentative_cost = current_cost + step_cost

            if tentative_cost >= g_score.get(neighbor, np.inf):
                continue

            parents[neighbor] = current
            g_score[neighbor] = tentative_cost
            priority = tentative_cost + h(neighbor, goal)
            heapq.heappush(open_heap, (priority, neighbor))

    raise RuntimeError('A* could not find a path from the start to the goal.')


def segment_is_clear(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: tuple[ObstacleSpec, ...],
    inflation_radius: float,
    sample_step: float,
) -> bool:
    segment = end - start
    length = float(np.linalg.norm(segment))

    if length < 1e-9:
        return not is_point_blocked(end, obstacles, inflation_radius)

    sample_count = max(2, int(np.ceil(length / sample_step)))
    for sample_index in range(sample_count + 1):
        ratio = sample_index / sample_count
        point = start + ratio * segment
        if is_point_blocked(point, obstacles, inflation_radius):
            return False

    return True


def append_safe_waypoints(
    output: list[np.ndarray],
    path_points: list[np.ndarray],
    start_index: int,
    end_index: int,
    obstacles: tuple[ObstacleSpec, ...],
    inflation_radius: float,
    sample_step: float,
) -> None:
    start_point = path_points[start_index]
    end_point = path_points[end_index]

    if end_index <= start_index + 1 or segment_is_clear(start_point, end_point, obstacles, inflation_radius, sample_step):
        if np.linalg.norm(output[-1] - end_point) > 1e-9:
            output.append(end_point)
        return

    middle_index = (start_index + end_index) // 2
    append_safe_waypoints(output, path_points, start_index, middle_index, obstacles, inflation_radius, sample_step)
    append_safe_waypoints(output, path_points, middle_index, end_index, obstacles, inflation_radius, sample_step)


def path_to_waypoints(
    path: list[tuple[int, int]],
    world: WorldModel,
    grid: OccupancyGrid,
    stride: int,
) -> list[np.ndarray]:
    if stride < 1:
        raise ValueError('Waypoint stride must be at least 1.')

    sampled_indices: list[int] = []

    for index in range(1, len(path) - 1):
        previous_cell = path[index - 1]
        current_cell = path[index]
        next_cell = path[index + 1]
        incoming_direction = (current_cell[0] - previous_cell[0], current_cell[1] - previous_cell[1])
        outgoing_direction = (next_cell[0] - current_cell[0], next_cell[1] - current_cell[1])

        if index % stride == 0 or incoming_direction != outgoing_direction:
            sampled_indices.append(index)

    path_points = [world.start.astype(float)]
    path_points.extend(grid_to_world(cell, grid) for cell in path[1:-1])
    path_points.append(world.goal.astype(float))

    candidate_indices = [0]
    candidate_indices.extend(sampled_indices)
    candidate_indices.append(len(path_points) - 1)

    inflation_radius = world.robot_radius + SAFETY_MARGIN
    sample_step = max(grid.resolution / 2.0, 0.05)
    waypoints = [path_points[0]]

    for previous_index, next_index in zip(candidate_indices, candidate_indices[1:]):
        append_safe_waypoints(
            waypoints,
            path_points,
            previous_index,
            next_index,
            world.obstacles,
            inflation_radius,
            sample_step,
        )

    return waypoints[1:]


class WaypointFollower:
    def __init__(self, waypoints: list[np.ndarray], reached_distance: float) -> None:
        if not waypoints:
            raise ValueError('Waypoint follower requires at least one waypoint.')

        self._waypoints = waypoints
        self._reached_distance = reached_distance
        self._index = 0

    @property
    def index(self) -> int:
        return self._index

    @property
    def is_finished(self) -> bool:
        return self._index >= len(self._waypoints) - 1

    def current_waypoint(self, position: np.ndarray) -> np.ndarray:
        while self._index < len(self._waypoints) - 1:
            distance = float(np.linalg.norm(self._waypoints[self._index] - position))
            if distance > self._reached_distance:
                break
            self._index += 1

        return self._waypoints[self._index]


def compute_action_from_state(state_xyt: np.ndarray, follower: WaypointFollower) -> np.ndarray:
    x_coord, y_coord, theta = state_xyt
    position = np.array([x_coord, y_coord], dtype=float)
    waypoint = follower.current_waypoint(position)

    delta = waypoint - position
    distance = float(np.linalg.norm(delta))
    target_heading = np.arctan2(delta[1], delta[0])
    heading_error = wrap_to_pi(target_heading - theta)

    angular_velocity = np.clip(2.0 * heading_error, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

    if follower.is_finished and distance < FINAL_GOAL_TOLERANCE:
        linear_velocity = 0.0
    elif abs(heading_error) > 0.9:
        linear_velocity = MIN_LINEAR_SPEED
    elif abs(heading_error) > 0.4:
        linear_velocity = 0.15
    else:
        linear_velocity = MAX_LINEAR_SPEED

    return np.array([[linear_velocity], [angular_velocity]], dtype=float)


def compute_action(robot, follower: WaypointFollower) -> np.ndarray:
    return compute_action_from_state(robot.state[:, 0], follower)


def plan_waypoints(map_path: str) -> tuple[WorldModel, OccupancyGrid, list[tuple[int, int]], list[np.ndarray]]:
    world = load_world(map_path)
    grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)
    start_cell, goal_cell = validate_start_and_goal(world, grid)
    path = astar_search(grid, start_cell, goal_cell)
    waypoints = path_to_waypoints(path, world, grid, WAYPOINT_STRIDE)
    return world, grid, path, waypoints


def run_simulation(map_path: str) -> None:
    world, _, path, waypoints = plan_waypoints(map_path)

    print(f'Planned {len(path)} grid steps and {len(waypoints)} waypoints from {map_path}.')

    if not waypoints or np.linalg.norm(world.start - world.goal) < 1e-9:
        print('The robot already starts at the goal.')
        return

    follower = WaypointFollower(waypoints, WAYPOINT_REACHED_DISTANCE)

    env = irsim.make(map_path)
    robot = env.robot_list[0]
    reached_goal = False
    failure_message: str | None = None

    try:
        for _ in range(MAX_STEPS):
            action = compute_action(robot, follower)
            env.step([action])
            env.render(0.05)

            if getattr(robot, 'arrive_flag', False):
                reached_goal = True
                break

            if env.done():
                if getattr(robot, 'collision_flag', False):
                    failure_message = 'The robot collided before reaching the goal.'
                elif getattr(robot, 'stop_flag', False):
                    failure_message = 'The robot stopped before reaching the goal.'
                else:
                    failure_message = 'The simulation ended before the robot reached the goal.'
                break
    finally:
        env.end()

    if not reached_goal:
        if failure_message is None:
            failure_message = f'The robot did not reach the goal within {MAX_STEPS} steps.'
        raise RuntimeError(failure_message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plan a route with A* and track it in irsim.')
    parser.add_argument('map_path', nargs='?', default='obstacle_harder.yaml', help='YAML world file to plan on.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_simulation(args.map_path)


if __name__ == '__main__':
    main()