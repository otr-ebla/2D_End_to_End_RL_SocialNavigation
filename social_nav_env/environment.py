"""Core 2D social navigation environment.

The environment models a simplified building floor plan populated with static walls
and moving humans. A disc-shaped robot navigates the space using 2D LiDAR
measurements and a goal location. The environment is designed to be compatible with
reinforcement learning pipelines.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    # Gymnasium provides the modern Gym API with explicit terminated/truncated flags.
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:  # pragma: no cover - handled gracefully.
    raise ModuleNotFoundError(
        "gymnasium is required to use SocialNavigationEnv. Install it with 'pip install gymnasium'."
    ) from exc


Vector = Tuple[float, float]


@dataclass
class EnvironmentConfig:
    """Configuration options for :class:`SocialNavigationEnv`.

    Attributes
    ----------
    dt:
        Simulation time step in seconds.
    robot_radius:
        Radius of the disc-shaped robot in meters.
    max_linear_speed:
        Maximum forward speed (m/s) applied to the robot.
    max_angular_speed:
        Maximum angular velocity (rad/s) applied to the robot.
    lidar_num_rays:
        Number of LiDAR beams uniformly distributed in :math:`[0, 2\pi)`.
    lidar_max_distance:
        Maximum measurable distance of each LiDAR beam (m).
    goal_tolerance:
        Distance threshold that marks the goal as reached.
    collision_penalty:
        Reward penalty applied when the robot collides with walls or humans.
    goal_reward:
        Reward granted once the robot reaches the goal region.
    step_penalty:
        Small penalty added every step to motivate efficiency.
    human_radius:
        Radius of the moving human agents.
    human_speed:
        Preferred linear speed of humans (m/s).
    num_humans:
        Number of human agents spawned in the environment.
    """

    dt: float = 0.1
    robot_radius: float = 0.25
    max_linear_speed: float = 0.8
    max_angular_speed: float = math.pi
    lidar_num_rays: int = 64
    lidar_max_distance: float = 6.0
    goal_tolerance: float = 0.3
    collision_penalty: float = -1.0
    goal_reward: float = 1.0
    step_penalty: float = -0.01
    human_radius: float = 0.25
    human_speed: float = 0.6
    num_humans: int = 3


@dataclass
class HumanAgent:
    """Simple holonomic agent that follows a looping sequence of waypoints."""

    radius: float
    speed: float
    waypoints: Sequence[Vector]
    position: Vector
    waypoint_index: int = 0

    def step(self, dt: float) -> None:
        if not self.waypoints:
            return

        target = self.waypoints[self.waypoint_index]
        direction = np.array(target) - np.array(self.position)
        distance = np.linalg.norm(direction)
        if distance < 1e-5:
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
            return

        direction /= distance
        travel = self.speed * dt
        if travel >= distance:
            self.position = target
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
        else:
            self.position = tuple(np.array(self.position) + direction * travel)


class SocialNavigationEnv(gym.Env):
    """2D indoor navigation environment with static walls and moving humans."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, config: EnvironmentConfig | None = None, seed: int | None = None):
        super().__init__()
        self.config = config or EnvironmentConfig()
        self._rng = np.random.default_rng(seed)

        self._build_layout()
        self._create_humans()

        obs_low = np.array([-np.inf, -np.inf, -math.pi, -np.inf, -np.inf] + [0.0] * self.config.lidar_num_rays)
        obs_high = np.array([np.inf, np.inf, math.pi, np.inf, np.inf] + [self.config.lidar_max_distance] * self.config.lidar_num_rays)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.robot_position: np.ndarray = np.zeros(2, dtype=float)
        self.robot_heading: float = 0.0
        self.goal_position: np.ndarray = np.zeros(2, dtype=float)
        self.humans: List[HumanAgent] = []
        self._figure = None
        self._axes = None

    # ------------------------------------------------------------------
    # Environment construction helpers
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Create static walls representing a building floor plan."""

        # Walls are represented as (x1, y1, x2, y2) line segments in meters.
        self._wall_segments: set[Tuple[float, float, float, float]] = set()

        def add_segment(x1: float, y1: float, x2: float, y2: float) -> None:
            segment = (x1, y1, x2, y2)
            reverse = (x2, y2, x1, y1)
            if reverse in self._wall_segments or segment in self._wall_segments:
                return
            self._wall_segments.add(segment)

        def add_rectangle(x_min: float, y_min: float, x_max: float, y_max: float) -> None:
            add_segment(x_min, y_min, x_max, y_min)
            add_segment(x_max, y_min, x_max, y_max)
            add_segment(x_max, y_max, x_min, y_max)
            add_segment(x_min, y_max, x_min, y_min)

        def add_wall_with_gaps(
            x1: float,
            y1: float,
            x2: float,
            y2: float,
            gaps: Sequence[Tuple[float, float]] | None = None,
        ) -> None:
            """Add an axis-aligned wall optionally leaving door openings."""

            gaps = sorted(gaps or [])
            if y1 == y2:
                if x2 < x1:
                    x1, x2 = x2, x1
                # Horizontal wall.
                current = x1
                for gap_start, gap_end in gaps:
                    if gap_start > current:
                        add_segment(current, y1, gap_start, y1)
                    current = max(current, gap_end)
                if current < x2:
                    add_segment(current, y1, x2, y1)
            elif x1 == x2:
                if y2 < y1:
                    y1, y2 = y2, y1
                # Vertical wall.
                current = y1
                for gap_start, gap_end in gaps:
                    if gap_start > current:
                        add_segment(x1, current, x1, gap_start)
                    current = max(current, gap_end)
                if current < y2:
                    add_segment(x1, current, x1, y2)
            else:
                raise ValueError("Walls must be axis-aligned")

        def add_room_with_door(
            x_min: float,
            y_min: float,
            x_max: float,
            y_max: float,
            door_side: str,
            door_center: float,
            door_width: float,
        ) -> None:
            """Create a room leaving an opening on the chosen wall."""

            half = door_width / 2.0
            if door_side == "south":
                add_wall_with_gaps(x_min, y_min, x_max, y_min, [(door_center - half, door_center + half)])
                add_segment(x_max, y_min, x_max, y_max)
                add_segment(x_max, y_max, x_min, y_max)
                add_segment(x_min, y_max, x_min, y_min)
            elif door_side == "north":
                add_segment(x_min, y_min, x_max, y_min)
                add_segment(x_max, y_min, x_max, y_max)
                add_wall_with_gaps(x_min, y_max, x_max, y_max, [(door_center - half, door_center + half)])
                add_segment(x_min, y_max, x_min, y_min)
            elif door_side == "east":
                add_segment(x_min, y_min, x_max, y_min)
                add_wall_with_gaps(x_max, y_min, x_max, y_max, [(door_center - half, door_center + half)])
                add_segment(x_max, y_max, x_min, y_max)
                add_segment(x_min, y_max, x_min, y_min)
            elif door_side == "west":
                add_segment(x_min, y_min, x_max, y_min)
                add_segment(x_max, y_min, x_max, y_max)
                add_segment(x_max, y_max, x_min, y_max)
                add_wall_with_gaps(x_min, y_min, x_min, y_max, [(door_center - half, door_center + half)])
            else:
                raise ValueError(f"Unsupported door side '{door_side}'")

        # Outer boundaries (18m x 12m area).
        add_rectangle(0.0, 0.0, 18.0, 12.0)

        # Horizontal main corridor (y = 4 .. 8) with doors to side rooms.
        add_wall_with_gaps(0.0, 4.0, 18.0, 4.0, gaps=[(3.5, 4.5), (8.5, 9.5), (13.5, 14.5)])
        add_wall_with_gaps(0.0, 8.0, 18.0, 8.0, gaps=[(2.0, 3.0), (7.0, 8.0), (12.0, 13.0), (16.0, 17.0)])

        # Vertical spine corridor (x = 7 .. 11) connecting north and south areas.
        add_wall_with_gaps(7.0, 0.0, 7.0, 12.0, gaps=[(1.5, 3.0), (5.0, 7.0), (9.0, 10.5)])
        add_wall_with_gaps(11.0, 0.0, 11.0, 12.0, gaps=[(2.5, 4.0), (6.0, 8.0), (10.0, 11.0)])

        # Suites in the south wing.
        add_room_with_door(0.5, 0.5, 3.5, 3.5, "north", 2.0, 1.2)
        add_room_with_door(3.8, 0.5, 6.5, 3.5, "north", 5.2, 1.0)
        add_room_with_door(11.5, 0.5, 14.5, 3.5, "north", 13.0, 1.5)
        add_room_with_door(14.8, 0.5, 17.5, 3.5, "north", 16.0, 1.0)

        # Meeting rooms along the spine corridor.
        add_room_with_door(7.2, 4.3, 10.8, 5.8, "west", 5.0, 1.0)
        add_room_with_door(7.2, 6.2, 10.8, 7.7, "east", 6.8, 1.2)

        # Offices in the north wing.
        add_room_with_door(0.5, 8.5, 4.0, 11.5, "south", 2.5, 1.4)
        add_room_with_door(4.3, 8.5, 6.8, 11.5, "south", 5.5, 1.0)
        add_room_with_door(11.5, 8.5, 14.5, 11.5, "south", 13.0, 1.2)
        add_room_with_door(14.8, 8.5, 17.5, 11.5, "south", 16.2, 1.2)

        # Convert to list for downstream logic.
        self.walls = list(self._wall_segments)

        # Define spawn zones for the robot and the goal.
        self.robot_spawn_regions = [
            ((8.0, 5.0), (10.0, 7.0)),  # central crossroads
            ((1.0, 1.0), (3.0, 3.0)),  # south-west office
            ((15.0, 9.0), (17.0, 11.0)),  # north-east office
        ]
        self.goal_regions = [
            ((1.0, 9.0), (3.5, 10.5)),
            ((12.0, 1.0), (16.5, 2.5)),
            ((8.0, 5.0), (10.0, 7.0)),
        ]

    def _create_humans(self) -> None:
        self.human_paths: List[List[Vector]] = [
            # Humans patrolling the main east-west corridor.
            [(4.5, 5.5), (13.5, 5.5), (13.5, 6.5), (4.5, 6.5)],
            # Human moving between the south offices.
            [(2.5, 1.0), (5.5, 1.0), (5.5, 3.0), (2.5, 3.0)],
            # Human traversing the north wing.
            [(12.5, 9.0), (16.0, 9.0), (16.0, 10.5), (12.5, 10.5)],
            # Human around the vertical spine intersection.
            [(8.5, 4.5), (8.5, 7.5), (9.5, 7.5), (9.5, 4.5)],
        self.walls: List[Tuple[float, float, float, float]] = []

        def add_rectangle(x_min: float, y_min: float, x_max: float, y_max: float) -> None:
            self.walls.extend(
                [
                    (x_min, y_min, x_max, y_min),
                    (x_max, y_min, x_max, y_max),
                    (x_max, y_max, x_min, y_max),
                    (x_min, y_max, x_min, y_min),
                ]
            )

        # Outer boundaries (a 14m x 10m area).
        add_rectangle(0.0, 0.0, 14.0, 10.0)

        # Corridors and rooms.
        # Central corridor along x-axis.
        add_rectangle(0.0, 4.5, 14.0, 5.5)
        # Side rooms separated by walls leaving doorways.
        # Left rooms.
        add_rectangle(0.0, 0.0, 4.0, 4.5)
        add_rectangle(0.0, 5.5, 4.0, 10.0)
        # Right rooms.
        add_rectangle(10.0, 0.0, 14.0, 4.5)
        add_rectangle(10.0, 5.5, 14.0, 10.0)

        # Remove wall segments to create doorways in corridor.
        self.doors: List[Tuple[float, float, float, float]] = [
            (4.0, 4.5, 4.0, 5.5),
            (10.0, 4.5, 10.0, 5.5),
            (7.0, 4.5, 7.0, 5.5),
            (7.0, 0.0, 7.0, 4.5),
            (7.0, 5.5, 7.0, 10.0),
        ]
        self.walls = [segment for segment in self.walls if segment not in self.doors]

        # Define spawn zones for the robot and the goal.
        self.robot_spawn_regions = [((1.0, 2.0), (3.0, 3.5)), ((11.0, 6.0), (13.0, 9.0))]
        self.goal_regions = [((12.0, 1.0), (13.0, 3.0)), ((1.0, 7.0), (2.0, 9.0))]

    def _create_humans(self) -> None:
        self.human_paths: List[List[Vector]] = [
            [(5.0, 4.75), (9.0, 4.75), (9.0, 5.25), (5.0, 5.25)],
            [(2.0, 1.0), (2.0, 3.5), (3.5, 3.5), (3.5, 1.0)],
            [(11.0, 7.0), (12.5, 7.0), (12.5, 9.0), (11.0, 9.0)],
        ]

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robot_position = self._sample_region(self.robot_spawn_regions)
        self.robot_heading = self._rng.uniform(-math.pi, math.pi)
        self.goal_position = self._sample_region(self.goal_regions)

        self.humans = []
        for i in range(self.config.num_humans):
            path = self.human_paths[i % len(self.human_paths)]
            start_index = self._rng.integers(0, len(path))
            agent = HumanAgent(
                radius=self.config.human_radius,
                speed=self.config.human_speed * self._rng.uniform(0.8, 1.2),
                waypoints=path,
                position=path[start_index],
                waypoint_index=start_index,
            )
            self.humans.append(agent)

        observation = self._get_observation()
        return observation, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        linear_cmd = action[0] * self.config.max_linear_speed
        angular_cmd = action[1] * self.config.max_angular_speed

        # Update robot pose.
        self.robot_heading = self._wrap_angle(self.robot_heading + angular_cmd * self.config.dt)
        heading_vector = np.array([math.cos(self.robot_heading), math.sin(self.robot_heading)])
        proposed_position = self.robot_position + heading_vector * linear_cmd * self.config.dt

        collision = self._check_collisions(proposed_position)
        if not collision:
            self.robot_position = proposed_position

        # Update humans.
        for human in self.humans:
            human.step(self.config.dt)

        observation = self._get_observation()
        reward = self._compute_reward(collision)
        terminated = self._is_goal_reached()
        if terminated:
            reward += self.config.goal_reward
        if collision:
            terminated = True
        truncated = False

        return observation, reward, terminated, truncated, {"collision": collision}

    # ------------------------------------------------------------------
    # Observation & reward
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        rel_goal = self.goal_position - self.robot_position
        observation = np.concatenate(
            [
                self.robot_position,
                np.array([self.robot_heading]),
                rel_goal,
                self._simulate_lidar(),
            ]
        )
        return observation.astype(np.float32)

    def _simulate_lidar(self) -> np.ndarray:
        num_rays = self.config.lidar_num_rays
        max_dist = self.config.lidar_max_distance
        angles = self.robot_heading + np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
        distances = np.full(num_rays, max_dist, dtype=float)

        for idx, angle in enumerate(angles):
            ray_origin = self.robot_position
            ray_dir = np.array([math.cos(angle), math.sin(angle)])

            for wall in self.walls:
                hit = self._ray_segment_intersection(ray_origin, ray_dir, wall)
                if hit is not None:
                    distances[idx] = min(distances[idx], hit)

            for human in self.humans:
                hit = self._ray_circle_intersection(ray_origin, ray_dir, human.position, human.radius)
                if hit is not None:
                    distances[idx] = min(distances[idx], hit)

        return distances.astype(np.float32)

    def _compute_reward(self, collision: bool) -> float:
        reward = self.config.step_penalty
        if collision:
            reward += self.config.collision_penalty
        distance = np.linalg.norm(self.goal_position - self.robot_position)
        reward += -0.05 * distance
        return float(reward)

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def _check_collisions(self, proposed_position: np.ndarray) -> bool:
        radius = self.config.robot_radius
        for wall in self.walls:
            if self._distance_point_to_segment(proposed_position, wall) <= radius:
                return True
        for human in self.humans:
            if np.linalg.norm(proposed_position - np.array(human.position)) <= radius + human.radius:
                return True
        return False

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _ray_segment_intersection(
        self, origin: np.ndarray, direction: np.ndarray, segment: Tuple[float, float, float, float]
    ) -> float | None:
        x1, y1, x2, y2 = segment
        v1 = origin - np.array([x1, y1])
        v2 = np.array([x2 - x1, y2 - y1])
        det = direction[0] * (-v2[1]) + direction[1] * v2[0]
        if abs(det) < 1e-8:
            return None
        t1 = (v2[0] * v1[1] - v2[1] * v1[0]) / det
        t2 = (direction[0] * v1[1] - direction[1] * v1[0]) / det
        if t1 >= 0 and 0 <= t2 <= 1:
            return t1
        return None

    def _ray_circle_intersection(
        self, origin: np.ndarray, direction: np.ndarray, center: Vector, radius: float
    ) -> float | None:
        oc = origin - np.array(center)
        b = 2 * np.dot(direction, oc)
        c = np.dot(oc, oc) - radius**2
        discriminant = b**2 - 4 * c
        if discriminant < 0:
            return None
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0
        hits = [t for t in (t1, t2) if t >= 0]
        if not hits:
            return None
        return min(hits)

    @staticmethod
    def _distance_point_to_segment(point: np.ndarray, segment: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = segment
        px, py = point
        dx = x2 - x1
        dy = y2 - y1
        if dx == dy == 0:
            return math.dist((px, py), (x1, y1))
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        closest = np.array([x1 + t * dx, y1 + t * dy])
        return float(np.linalg.norm(point - closest))

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return (theta + math.pi) % (2 * math.pi) - math.pi

    def _is_goal_reached(self) -> bool:
        return np.linalg.norm(self.goal_position - self.robot_position) <= self.config.goal_tolerance

    def _sample_region(self, regions: Iterable[Tuple[Vector, Vector]]) -> np.ndarray:
        region = regions[self._rng.integers(0, len(regions))]
        (x_min, y_min), (x_max, y_max) = region
        x = self._rng.uniform(x_min, x_max)
        y = self._rng.uniform(y_min, y_max)
        return np.array([x, y], dtype=float)

    # ------------------------------------------------------------------
    # Rendering (optional)
    # ------------------------------------------------------------------

    def render(self):
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
            raise ModuleNotFoundError(
                "matplotlib is required for rendering. Install it with 'pip install matplotlib'."
            ) from exc

        if self._figure is None or self._axes is None:
            plt.ion()
            self._figure, self._axes = plt.subplots(figsize=(8, 5))
        ax = self._axes
        ax.clear()
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 12)
        plt.figure(figsize=(6, 4))
        ax = plt.gca()
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")

        for wall in self.walls:
            x1, y1, x2, y2 = wall
            ax.plot([x1, x2], [y1, y2], "k-", linewidth=2)
            ax.plot([x1, x2], [y1, y2], "k-")

        for human in self.humans:
            circle = plt.Circle(human.position, human.radius, color="orange", alpha=0.7)
            ax.add_patch(circle)

        robot = plt.Circle(self.robot_position, self.config.robot_radius, color="blue", alpha=0.7)
        ax.add_patch(robot)
        ax.arrow(
            self.robot_position[0],
            self.robot_position[1],
            0.6 * math.cos(self.robot_heading),
            0.6 * math.sin(self.robot_heading),
            head_width=0.25,
            0.5 * math.cos(self.robot_heading),
            0.5 * math.sin(self.robot_heading),
            head_width=0.2,
            color="blue",
        )

        goal = plt.Circle(self.goal_position, self.config.goal_tolerance, color="green", alpha=0.4)
        ax.add_patch(goal)

        ax.set_title("2D Social Navigation")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, which="both", linestyle="--", linewidth=0.3)
        self._figure.tight_layout()
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
        plt.show(block=False)


def keyboard_control(
    env: SocialNavigationEnv,
    *,
    linear_step: float = 0.2,
    angular_step: float = 0.2,
    max_steps: int | None = None,
) -> None:
    """Interactively drive the robot using arrow keys.

    Parameters
    ----------
    env:
        Instance of :class:`SocialNavigationEnv` to control. The environment is
        reset at the beginning of the session and whenever the user presses ``r``.
    linear_step:
        Increment applied to the normalized forward command each time the up or
        down arrow is pressed. The resulting command is clamped to ``[-1, 1]``.
    angular_step:
        Increment applied to the normalized angular command each time the left or
        right arrow is pressed. The resulting command is clamped to ``[-1, 1]``.
    max_steps:
        Optional cap on the number of simulation steps per episode. When reached,
        the episode ends and the user can restart or quit.
    """

    if not isinstance(env, SocialNavigationEnv):
        raise TypeError("keyboard_control expects a SocialNavigationEnv instance")

    import curses

    def clamp(value: float) -> float:
        return float(max(-1.0, min(1.0, value)))

    def control_loop(stdscr: "curses._CursesWindow") -> None:  # type: ignore[name-defined]
        curses.curs_set(0)
        stdscr.nodelay(True)
        linear_cmd = 0.0
        angular_cmd = 0.0
        steps = 0
        env.reset()
        env.render()
        episode_finished = False

        while True:
            key = stdscr.getch()
            while key != -1:
                if key == curses.KEY_UP:
                    linear_cmd = clamp(linear_cmd + linear_step)
                elif key == curses.KEY_DOWN:
                    linear_cmd = clamp(linear_cmd - linear_step)
                elif key == curses.KEY_LEFT:
                    angular_cmd = clamp(angular_cmd + angular_step)
                elif key == curses.KEY_RIGHT:
                    angular_cmd = clamp(angular_cmd - angular_step)
                elif key in (ord(" "),):
                    linear_cmd = 0.0
                    angular_cmd = 0.0
                elif key in (ord("r"), ord("R")):
                    env.reset()
                    env.render()
                    linear_cmd = 0.0
                    angular_cmd = 0.0
                    steps = 0
                    episode_finished = False
                elif key in (ord("q"), ord("Q")):
                    return
                key = stdscr.getch()

            stdscr.erase()
            stdscr.addstr(0, 0, "Arrow keys adjust linear/angular commands (space to stop, r to reset, q to quit)")
            stdscr.addstr(1, 0, f"Linear command: {linear_cmd:+.2f}    Angular command: {angular_cmd:+.2f}")
            stdscr.addstr(2, 0, f"Steps: {steps}")
            stdscr.refresh()

            if episode_finished:
                stdscr.addstr(3, 0, "Episode finished. Press 'r' to reset or 'q' to quit.")
                stdscr.refresh()
                time.sleep(0.1)
                continue

            action = np.array([linear_cmd, angular_cmd], dtype=np.float32)
            _, reward, terminated, truncated, _ = env.step(action)
            env.render()
            steps += 1

            stdscr.addstr(3, 0, f"Last reward: {reward:+.3f}")
            stdscr.refresh()

            if terminated or truncated:
                episode_finished = True
            elif max_steps is not None and steps >= max_steps:
                episode_finished = True

            if episode_finished:
                stdscr.addstr(4, 0, "Episode finished. Press 'r' to reset or 'q' to quit.")
                stdscr.refresh()
            time.sleep(env.config.dt)

    curses.wrapper(control_loop)
        plt.tight_layout()
        plt.show()

