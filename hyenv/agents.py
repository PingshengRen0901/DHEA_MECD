"""Reduced translational dynamics for the particle simulator (paper Eq. 4)."""
import numpy as np


class MovingAgent:
    def __init__(self, delta_t=0.005, mass=1.0, linear_drag=0.1,
                 quadratic_drag=0.1, max_speed=2.0):
        self.delta_t = delta_t
        self.mass = mass
        self.linear_drag = linear_drag
        self.quadratic_drag = quadratic_drag
        self.max_speed = max_speed

    def reset(self, position, yaw=0.0, pitch=0.0):
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.zeros(3)
        self.theta, self.phi = float(yaw), float(pitch)

    @property
    def speed(self):
        return float(np.linalg.norm(self.velocity))

    @property
    def direction(self):
        return np.array([np.cos(self.phi) * np.cos(self.theta),
                         np.cos(self.phi) * np.sin(self.theta), np.sin(self.phi)])

    def advance(self, action, parameters, current, disturbance,
                max_acceleration, max_yaw_rate, max_pitch_rate, max_braking):
        acceleration = braking = 0.0
        if action == 0:
            acceleration = max_acceleration * (parameters[0] + 1) / 2
        elif action == 1:
            self.theta = (self.theta + max_yaw_rate * parameters[1] * self.delta_t) % (2 * np.pi)
        elif action == 2:
            self.phi = np.clip(self.phi + max_pitch_rate * parameters[2] * self.delta_t,
                               -np.pi / 2, np.pi / 2)
        elif action == 3:
            braking = max_braking * (parameters[3] + 1) / 2
        relative = self.velocity - current
        drag = -self.linear_drag * relative - self.quadratic_drag * np.linalg.norm(relative) * relative
        force = self.mass * acceleration * self.direction + drag + disturbance
        self.velocity += self.delta_t * force / self.mass
        speed = self.speed
        if speed > 0:
            self.velocity *= max(0.0, speed - braking * self.delta_t) / speed
        if self.speed > self.max_speed:
            self.velocity *= self.max_speed / self.speed
        self.position += self.delta_t * self.velocity
