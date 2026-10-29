"""Gymnasium environment registration (Moving-v0 retained for existing users)."""
from gymnasium.envs.registration import register, registry
from hyenv.environment import MovingEnv

for env_id in ("Moving-v0", "AUVTracking-v0"):
    if env_id not in registry:
        register(id=env_id, entry_point="hyenv.environment:MovingEnv")

__all__ = ["MovingEnv"]
