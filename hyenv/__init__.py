# my envs
from gym.envs.registration import register
from hyenv.environment import MovingEnv
from hyenv.environment import SlidingEnv


#注册自定义Gym环境
register(
    id='Moving-v0',
    entry_point='hyenv.environment:MovingEnv',
)
register(
    id='Sliding-v0',
    entry_point='hyenv.environment:SlidingEnv',
)