"""Core 2D social navigation environment.

The environment models a simplified building floor plan populated with static walls
and moving humans. A disc-shaped robot navigates the space using 2D LiDAR
measurements and a goal location. The environment is designed to be compatible with
reinforcement learning pipelines.
"""

from __future__ import annotations

import math
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

    # ------------------------------------------------------------------
    # Environment construction helpers
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Create static walls representing a building floor plan."""

        # Walls are represented as (x1, y1, x2, y2) line segments in meters.
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

        plt.figure(figsize=(6, 4))
        ax = plt.gca()
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")

        for wall in self.walls:
            x1, y1, x2, y2 = wall
            ax.plot([x1, x2], [y1, y2], "k-")

        for human in self.humans:
            circle = plt.Circle(human.position, human.radius, color="orange", alpha=0.7)
            ax.add_patch(circle)

        robot = plt.Circle(self.robot_position, self.config.robot_radius, color="blue", alpha=0.7)
        ax.add_patch(robot)
        ax.arrow(
            self.robot_position[0],
            self.robot_position[1],
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
        plt.tight_layout()
        plt.show()

