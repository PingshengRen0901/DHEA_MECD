"""Three-dimensional, sequential multi-target acquisition simulator.

The paper does not specify target assignment or hydrodynamic coefficients.
Those choices are exposed here and documented in README.md.
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import ObservationLayout
from hyenv.agents import MovingAgent
from hyenv.disturbances import OceanCurrent, UnderwaterNoise
from hyenv.observations import appendix_observation

ACCELERATE, TURN, PITCH, BREAK = range(4)


@dataclass
class RewardConfig:
    distance: float = 10.0
    goal: float = 10.0
    collision: float = 5.0
    boundary: float = 2.0


class MovingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, n_targets=3, n_obstacles=2, max_step=800, delta_t=0.005,
                 field_size=1.0, target_radius=0.1, obstacle_radius=0.08,
                 agent_radius=0.025, target_speed=0.05, obstacle_speed=0.02,
                 current_strength=0.1, noise_std=0.01, max_speed=2.0,
                 max_acceleration=0.5, max_turn=np.pi / 2, max_pitch=np.pi / 4,
                 max_braking=1.0, reward_config=None, render_mode=None, observation_dim=None,
                 propulsion_weight=1.0, steering_weight=1.0, energy_reference=1.0,
                 length_scale_m=None):
        if max_step < 1 or delta_t <= 0 or field_size <= 0 or target_radius <= 0:
            raise ValueError("Invalid time or workspace configuration")
        if min(target_speed, obstacle_speed, current_strength, noise_std) < 0 or max_speed <= 0:
            raise ValueError("Speeds, current and noise must be nonnegative; max_speed positive")
        if obstacle_radius + agent_radius >= field_size or target_radius >= field_size:
            raise ValueError("Object radii must fit inside the workspace")
        if energy_reference <= 0 or min(propulsion_weight, steering_weight) < 0:
            raise ValueError("Energy reference must be positive and weights nonnegative")
        if length_scale_m is not None and length_scale_m <= 0:
            raise ValueError("length_scale_m must be positive when supplied")
        self.layout = ObservationLayout(n_targets, n_obstacles, observation_dim)
        self.propulsion_weight, self.steering_weight = propulsion_weight, steering_weight
        self.energy_reference, self.length_scale_m = energy_reference, length_scale_m
        self.n_targets, self.n_obstacles = n_targets, n_obstacles
        self.max_step, self.delta_t = max_step, delta_t
        self.field_size, self.target_radius = field_size, target_radius
        self.obstacle_radius, self.agent_radius = obstacle_radius, agent_radius
        self.target_speed, self.obstacle_speed = target_speed, obstacle_speed
        self.current_strength, self.noise_std = current_strength, noise_std
        self.max_acceleration, self.max_turn, self.max_pitch = max_acceleration, max_turn, max_pitch
        self.max_braking = max_braking
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode
        self.agent = MovingAgent(delta_t=delta_t, max_speed=max_speed)
        self.action_space = spaces.Tuple((spaces.Discrete(4), spaces.Box(-1.0, 1.0, (4,), np.float32)))
        self.observation_space = spaces.Box(0.0, 1.0, (self.layout.state_dim,), np.float32)
        self._finished = True
        self._figure = None

    def _positions(self, count, margin=0.15):
        return self.np_random.uniform(-self.field_size + margin, self.field_size - margin, (count, 3))

    def _velocities(self, count, speed):
        v = self.np_random.normal(size=(count, 3))
        return speed * v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-10)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.time = 0.0
        self.yaw_rate = 0.0
        self.agent.reset(self._positions(1, self.agent_radius)[0],
                         self.np_random.uniform(0, 2 * np.pi), self.np_random.uniform(-0.3, 0.3))
        self.targets = self._positions(self.n_targets, self.target_radius)
        self.target_velocities = self._velocities(self.n_targets, self.target_speed)
        self.obstacles = self._positions(self.n_obstacles, self.obstacle_radius)
        # Avoid initial collisions without allowing an unbounded rejection loop.
        for i in range(self.n_obstacles):
            for _ in range(1000):
                if np.linalg.norm(self.obstacles[i] - self.agent.position) > self.agent_radius + self.obstacle_radius:
                    break
                self.obstacles[i] = self._positions(1, self.obstacle_radius)[0]
            else:
                raise RuntimeError("Could not sample a collision-free initial state")
        self.obstacle_velocities = self._velocities(self.n_obstacles, self.obstacle_speed)
        self.captured = np.zeros(self.n_targets, dtype=bool)
        self.current_model = OceanCurrent(self.np_random, self.field_size, self.current_strength)
        self.noise_model = UnderwaterNoise(self.np_random, self.noise_std)
        self.current = self.current_model.velocity(self.agent.position, self.time)
        self.noise = self.noise_model.sample(self.agent.speed, np.linalg.norm(self.current))
        self.trajectory_length = self.energy_cost = 0.0
        self._finished = False
        return self.get_state(), self._info()

    @property
    def distances(self):
        return np.linalg.norm(self.targets - self.agent.position, axis=1)

    @property
    def active_target(self):
        return int(np.argmin(np.where(self.captured, np.inf, self.distances)))

    @property
    def distance(self):
        return float(self.distances[self.active_target])

    def _move_objects(self, positions, velocities, radius):
        positions += velocities * self.delta_t
        limit = self.field_size - radius
        for axis in range(3):
            outside = np.abs(positions[:, axis]) > limit
            velocities[outside, axis] *= -1
        np.clip(positions, -limit, limit, out=positions)

    def check_obstacle_collison(self):
        return bool(np.any(np.linalg.norm(self.obstacles - self.agent.position, axis=1)
                           <= self.agent_radius + self.obstacle_radius))

    def step(self, action):
        if self._finished:
            raise RuntimeError("Call reset() before stepping a finished episode")
        discrete, parameters = action
        parameters = np.asarray(parameters, dtype=np.float32)
        if not self.action_space[0].contains(discrete) or parameters.shape != (4,) or not np.isfinite(parameters).all():
            raise ValueError("Action must be (discrete_id in [0,3], four finite parameters)")
        parameters = np.clip(parameters, -1, 1)
        target = self.active_target
        previous_distance = self.distances[target]
        previous_position = self.agent.position.copy()
        # n(t) affects actuation as well as being observed; no unused noise appendices.
        disturbance = np.full(3, np.clip(self.noise.sum(), -0.1, 0.1))
        self.yaw_rate = self.max_turn * float(parameters[1]) if discrete == TURN else 0.0
        self.agent.advance(discrete, parameters, self.current, disturbance,
                           self.max_acceleration, self.max_turn, self.max_pitch, self.max_braking)
        self._move_objects(self.targets, self.target_velocities, self.target_radius)
        self._move_objects(self.obstacles, self.obstacle_velocities, self.obstacle_radius)
        self.current_step += 1
        self.time = self.current_step * self.delta_t
        self.current = self.current_model.velocity(self.agent.position, self.time)
        self.noise = self.noise_model.sample(self.agent.speed, np.linalg.norm(self.current))
        collision = self.check_obstacle_collison()
        boundary = bool(np.any(np.abs(self.agent.position) > self.field_size))
        # A collision/boundary event cannot also earn a target bonus.
        new_captures = (~self.captured) & (self.distances < self.target_radius)
        if collision or boundary:
            new_captures[:] = False
        self.captured |= new_captures
        cfg = self.reward_config
        components = {
            "distance": cfg.distance * float(previous_distance - self.distances[target]),
            "goal": cfg.goal * int(new_captures.sum()),
            "collision": -cfg.collision * int(collision),
            "boundary": -cfg.boundary * int(boundary),
        }
        success = bool(self.captured.all())
        terminated = success or collision or boundary
        truncated = self.current_step >= self.max_step and not terminated
        self._finished = terminated or truncated
        self.trajectory_length += float(np.linalg.norm(self.agent.position - previous_position))
        # Unit-weight command cost (Table I lambda_p=lambda_s=1); only active command contributes.
        magnitude = (parameters[discrete] + 1) / 2 if discrete in (ACCELERATE, BREAK) else parameters[discrete]
        weight = self.propulsion_weight if discrete in (ACCELERATE, BREAK) else self.steering_weight
        self.energy_cost += float(weight * magnitude ** 2 * self.delta_t / self.energy_reference)
        info = self._info(collision, boundary)
        info["reward_components"] = components
        return self.get_state(), float(sum(components.values())), terminated, truncated, info

    def get_state(self):
        if self.layout.observation_dim is not None:
            return appendix_observation(self)
        # Fixed finite bounds preserve meaning across episodes; heavy-tailed noise uses arctan.
        position = lambda x: np.clip((np.asarray(x) / self.field_size + 1) / 2, 0, 1)
        space = np.concatenate([position(self.agent.position), position(self.targets).ravel(),
                                np.clip(self.distances / (2 * np.sqrt(3) * self.field_size), 0, 1),
                                self.captured.astype(float)])
        motion = np.concatenate([(self.agent.velocity / self.agent.max_speed + 1) / 2,
                                 (np.array([np.sin(self.agent.theta), np.cos(self.agent.theta),
                                            np.sin(self.agent.phi), np.cos(self.agent.phi)]) + 1) / 2,
                                 [min(self.current_step / self.max_step, 1.0)]])
        barrier = np.concatenate([np.clip((self.current / max(self.current_strength, 1e-8) + 1) / 2, 0, 1),
                                  position(self.obstacles).ravel()])
        noise = 0.5 + np.arctan(self.noise / max(self.noise_std, 1e-8)) / np.pi
        return np.concatenate([space, motion, barrier, noise]).astype(np.float32)

    def _info(self, collision=False, boundary=False):
        return {"success": bool(self.captured.all()), "captured_targets": int(self.captured.sum()),
                "collision": collision, "boundary": boundary,
                "trajectory_length": self.trajectory_length, "energy_cost": self.energy_cost}

    def render(self):
        import matplotlib.pyplot as plt
        if self._figure is None:
            self._figure = plt.figure(figsize=(6, 6))
        self._figure.clear()
        ax = self._figure.add_subplot(111, projection="3d")
        ax.scatter(*self.agent.position, c="blue", label="AUV")
        for i, point in enumerate(self.targets):
            ax.scatter(*point, c="gray" if self.captured[i] else "green", marker="*")
        if self.n_obstacles:
            ax.scatter(*self.obstacles.T, c="red", s=80, label="Obstacles")
        ax.set(xlim=(-self.field_size, self.field_size), ylim=(-self.field_size, self.field_size),
               zlim=(-self.field_size, self.field_size), title=f"Step {self.current_step}")
        self._figure.canvas.draw()
        if self.render_mode == "rgb_array":
            return np.asarray(self._figure.canvas.buffer_rgba())[..., :3].copy()
        plt.pause(0.001)

    def close(self):
        if self._figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self._figure)
            self._figure = None
