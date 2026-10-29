"""RBF currents and four underwater noise sources, paper Eqs. (1)-(11)."""
import numpy as np


class OceanCurrent:
    def __init__(self, rng, field_size, strength=0.1, bases=6):
        self.centers = rng.uniform(-field_size, field_size, (bases, 3))
        self.scales = rng.uniform(0.4, 1.0, bases) * field_size
        # The convex bound keeps each component within +/- strength.
        self.weights = rng.uniform(-1, 1, (bases, 3)) * strength / bases
        self.phase = rng.uniform(0, 2 * np.pi, (bases, 3))

    def velocity(self, position, time):
        phi = np.exp(-np.sum((position - self.centers) ** 2, axis=1) / (2 * self.scales ** 2))
        theta = self.weights * np.cos(0.5 * time + self.phase)
        return phi @ theta


class UnderwaterNoise:
    def __init__(self, rng, std=0.01, alpha=1.5, rho=0.9, geo_beta=1.0):
        self.rng, self.std, self.alpha, self.rho = rng, std, alpha, rho
        self.turbulence = 0.0
        # Fractional integration: |H(w)|^2 approximately w^-beta at low frequencies.
        self.geo_filter = np.ones(128)
        d = geo_beta / 2
        for k in range(1, len(self.geo_filter)):
            self.geo_filter[k] = self.geo_filter[k - 1] * (k - 1 + d) / k
        self.geo_filter /= np.linalg.norm(self.geo_filter)
        self.geo_history = rng.normal(size=len(self.geo_filter))

    def sample(self, speed, current_speed):
        vehicle = self.rng.normal(0, self.std * max(speed, 0.0))
        # Chambers-Mallows-Stuck sampler for symmetric alpha-stable noise.
        u = self.rng.uniform(-np.pi / 2 + 1e-10, np.pi / 2 - 1e-10)
        w = max(self.rng.exponential(), 1e-10)
        a = self.alpha
        biological = self.std * np.sin(a * u) / np.cos(u) ** (1 / a)
        biological *= (np.cos((1 - a) * u) / w) ** ((1 - a) / a)
        self.geo_history = np.roll(self.geo_history, 1)
        self.geo_history[0] = self.rng.normal()
        geological = self.std * (self.geo_filter @ self.geo_history)
        self.turbulence = self.rho * self.turbulence + self.std * current_speed * self.rng.normal()
        return np.array([vehicle, biological, geological, self.turbulence])
