import numpy as np
from typing import Tuple
from typing import Optional
from typing import List
from collections import namedtuple
import pyglet
import gym
from gym import spaces
from gym.utils import seeding
# 兼容gym和gymnasium的logger API
try:
    gym.logger.set_level(40)  # gym的API
except AttributeError:
    gym.logger.min_level = 40  # gymnasium的API
from hyenv.agents import BaseAgent
from hyenv.agents import MovingAgent
from hyenv.agents import SlidingAgent


# Action Id
ACCELERATE = 0
TURN = 1
PITCH = 2  # 新增俯仰动作
BREAK = 3

Target = namedtuple('Target', ['x', 'y', 'z', 'radius'])  # 添加z坐标

class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        self.label.draw()


# 动作空间:

class Action:
    def __init__(self, id_: int, parameters: list):
        self.id = id_
        self.parameters = parameters

    @property
    def parameter(self) -> float:
        # 刹车动作不需要参数
        if self.id == BREAK:
            return 0.0
        # 其他动作返回对应参数
        # 安全处理：确保索引在范围内
        if self.id < len(self.parameters):
            return self.parameters[self.id]
        else:
            return 0.0  # 默认值


class Obstacle:
    def __init__(self, x: float, y: float, z: float, r: float, speed: float):  # 添加z坐标
        self.x = x
        self.y = y
        self.z = z  # 添加z坐标
        self.r = r
        self.vx = speed
        self.vy = speed
        self.vz = speed # z方向速度设为0

    def update(self, delta_t: float) -> None:
        self.x += self.vx * delta_t
        self.y += self.vy * delta_t
        self.z += self.vz * delta_t  # z方向移动



    def update(self, delta_t: float) -> None:
        self.x += self.vx * delta_t
        self.y += self.vy * delta_t
        self.z += self.vz * delta_t



 # 基础环境类
class BaseEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    """"
    Gym environment parent class.
    """
    def __init__(
            self,
            seed: Optional[int] = None,
            max_turn: float = np.pi/2,
            max_pitch: float = np.pi / 4,  # 添加最大俯仰角参数
            max_acceleration: float = 0.5,
            delta_t: float = 0.005,
            max_step: int = 200,
            penalty: float = 0.001,
            break_value: float = 0.1,


            n_obstacles: int = 2,
            obstacles_radius: float = 0.15,
            obstacle_speed: float = 1.5,


    ):

        # Agent Parameters
        self.max_turn = max_turn
        self.max_pitch = max_pitch
        self.max_acceleration = max_acceleration
        self.break_value = break_value

        # Environment Parameters
        self.delta_t = delta_t
        self.max_step = max_step
        self.field_size = 1.0
        self.target_radius = 0.1
        self.penalty = penalty

        # Initialization
        self.seed(seed)
        self.target = None
        self.viewer = None
        self.current_step = None
        self.agent = BaseAgent(break_value=break_value, delta_t=delta_t)
        self.mid_target_1 = None
        ""
        self.obstacles = []
        self.n_obstacles = n_obstacles
        self.obstacle_radius = obstacles_radius
        self.obstacle_speed = obstacle_speed
        ""

        #洋流信息

        self.current_x = 0.0  # 横向洋流初始值
        self.current_y = 0.1  # 纵向固定洋流初始值
        self.current_z = 0.0  # z向洋流初始值
        self.time = 0.0  # 时间变量
        self.current_amplitude = 0.02  # 洋流幅值
        self.current_omega = 1.0  # 洋流角频率

        # 高斯噪声标准差（作为状态中额外的4维噪声）
        self.noise_std = 0.01  # 默认值，可根据需要调整

        parameters_min = np.array([0, -1, -1])
        parameters_max = np.array([1, +1, +1])
        self.action_space = spaces.Tuple((
            spaces.Discrete(4),  # 现在有4个动作：加速、转弯、刹车、俯仰
            spaces.Box(parameters_min, parameters_max, dtype=np.float32)
        ))
        # 观测空间维度: 当前仅使用9维基础状态
        obs_dim = 9
        obs_low = -np.inf * np.ones(obs_dim, dtype=np.float32)
        obs_high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def seed(self, seed: Optional[int] = None) -> list:
        self.np_random, seed = seeding.np_random(seed)  # noqa
        return [seed]

    def reset(self, seed=None, options=None):

        self.current_step = 0

        """目的地"""
        limit = 0.5
        low = [limit, limit, self.target_radius]
        high = [limit, limit, self.target_radius]


        self.target = Target(
        self.np_random.uniform(0.5, 0.5),  # x
        self.np_random.uniform(0.5, 0.5),  # y
        0.0,  # z坐标初始化为0（保持二维平面）
        self.target_radius
    )


        self.obstacles = []
        for _ in range(self.n_obstacles): #生成两种障碍物
            if _ == 0:
                x = np.random.uniform(self.obstacle_radius, 0.5)  # [0.15, 0.5]
                y = np.random.uniform(self.obstacle_radius, 0.5)
                z = np.random.uniform(self.obstacle_radius, 0.5)  # 与x,y类似的z坐标范围
            if _ == 1:
                # self.obstacle_radius += 0.1
                # 绿色
                x = np.random.uniform(0, 1 - self.obstacle_radius)
                y = np.random.uniform(0, 1 - self.obstacle_radius)
                z = np.random.uniform(0, 1 - self.obstacle_radius)  # 与x,y类似的z坐标范围

            obstacle = Obstacle(x, y, z, self.obstacle_radius, self.obstacle_speed)
            self.obstacles.append(obstacle)

        low = [-self.field_size, -self.field_size, 0, 0]  # x, y, z, theta 下界（z固定为0）
        high = [self.field_size, self.field_size, 0, 2 * np.pi]  # 上界
        x, y, z, theta = self.np_random.uniform(low, high)
        phi = 0.0  # 初始俯仰角为0，保持在水平面
        self.agent.reset(x, y, z, theta, phi)

        # 确保洋流状态在reset后可用
        self.current_x = 0.0
        self.current_y = 0.1
        self.current_z = 0.0
        self.time = 0.0

        return self.get_state()

    def step(self, raw_action: Tuple[int, list]) -> Tuple[list, float, bool, dict]:
        action = Action(*raw_action)
        last_distance = self.distance
        self.current_step += 1
        self.update_obstacles()

        if action.id == TURN:
            rotation = self.max_turn * max(min(action.parameter, 1), -1)
            self.agent.turn(rotation)
        elif action.id == ACCELERATE:
            acceleration = self.max_acceleration * max(min(action.parameter, 1), 0)
            self.agent.accelerate(acceleration)
        elif action.id == PITCH:
            pitch_change = self.max_pitch * max(min(action.parameter, 1), -1)
            self.agent.pitch(pitch_change)
        elif action.id == BREAK:
            self.agent.break_()


        # 🌊 洋流推动智能体
        self.time += self.delta_t

        # 动态洋流
        self.current_x = self.current_amplitude * np.sin(self.current_omega * self.time)
        self.current_y = 0.1  # 固定纵向洋流
        self.current_z = 0.0  # z方向洋流为0


        # 应用洋流
        self.agent.x += self.current_x * self.delta_t
        self.agent.y += self.current_y * self.delta_t
        self.agent.z += self.current_z * self.delta_t

        if self.distance < self.target_radius and self.agent.speed == 0:
            reward = self.get_reward(last_distance, True)
            done = True
        elif (abs(self.agent.x) > self.field_size or
              abs(self.agent.y) > self.field_size or
              abs(self.agent.z) > self.field_size or
              self.current_step >= self.max_step):
            reward = -1
            done = True
        elif self.check_obstacle_collison():
            reward = -1
            done = True
        else:
            reward = self.get_reward(last_distance)
            done = False

        return self.get_state(), reward, done, {}

    def get_state(self) -> np.ndarray:
        # 基础状态向量 (9维)
        state = np.array([
            self.agent.x,
            self.agent.y,
            self.agent.z,  # 添加z坐标
            # self.agent.speed,
            # np.cos(self.agent.theta),
            # np.sin(self.agent.theta),
            # np.cos(self.agent.phi),
            # np.sin(self.agent.phi),
            self.target.x,
            self.target.y,
            self.target.z,  # 目标z坐标
            self.distance,
            0 if self.distance > self.target_radius else 1,
            self.current_step / self.max_step,
            # self.current_x,
            # self.current_y,
            # self.current_z,  # z方向洋流
        ], dtype=np.float32)

        # # 额外高斯噪声（4维）
        # noise = self.np_random.normal(0.0, self.noise_std, size=13).astype(np.float32)

        # # 障碍物相对位置添加z分量
        # obstacles_relative = []
        # for obs in self.obstacles:
        #     dx = obs.x - self.agent.x
        #     dy = obs.y - self.agent.y
        #     dz = obs.z - self.agent.z
        #     obstacles_relative.extend([dx, dy, dz])

        # # 确保障碍物数据是numpy数组
        # obstacles_relative = np.array(obstacles_relative, dtype=np.float32)

        # 合并所有状态组件
        # state_with_extra = np.concatenate([
        #     np.array(state, dtype=np.float32), 
        #     noise, 
        #     obstacles_relative
        # ])

        return state



    # def get_reward(self, last_distance: float, goal: bool = False) -> float:
    #     return last_distance - self.distance - self.penalty + (1 if goal else 0)
    def get_reward(self, last_distance: float, goal: bool = False) -> float:
        """改进的奖励函数"""
        current_distance = self.distance

        # 基础距离奖励 - 鼓励接近目标
        distance_reward = (last_distance - current_distance) * 10

        # 到达目标奖励
        goal_reward = 10 if goal else 0

        # 碰撞惩罚
        collision_penalty = -5 if self.check_obstacle_collison() else 0

        # 边界惩罚
        boundary_penalty = -2 if (abs(self.agent.x) > 0.9 or
                                 abs(self.agent.y) > 0.9 or
                                 abs(self.agent.z) > 0.9) else 0

        total_reward = (distance_reward +  + goal_reward +
                       collision_penalty + boundary_penalty)

        return total_reward
    @property
    def distance(self) -> float:
        # 三维距离计算
        return np.sqrt(
            (self.agent.x - self.target.x)**2 + 
            (self.agent.y - self.target.y)**2 +
            (self.agent.z - self.target.z)**2
        )

    @staticmethod
    def get_distance(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
        """计算三维空间中的欧几里得距离"""
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    def check_obstacle_collison(self) -> bool:
        for obstacle in self.obstacles:
            # 三维距离计算
            distance = np.sqrt(
                (self.agent.x - obstacle.x)**2 + 
                (self.agent.y - obstacle.y)**2 +
                (self.agent.z - obstacle.z)**2
            )
            if distance <= (0.05 + obstacle.r):
                return True
        return False

    def update_obstacles(self) -> None:
        for obstacle in self.obstacles:
            obstacle.update(self.delta_t)
            if obstacle.x < obstacle.r or obstacle.x > (1 - obstacle.x):
                obstacle.vx = -obstacle.vx
            if obstacle.y < obstacle.r or obstacle.y > (1 - obstacle.y):
                obstacle.vy = -obstacle.vy
            if obstacle.z < obstacle.r or obstacle.z > (1 - obstacle.z):
                obstacle.vz = -obstacle.vz
    ""


    def render(self, mode='human'):
        screen_width = 400
        screen_height = 400
        unit_x = screen_width / 2
        unit_y = screen_height / 2
        agent_radius = 0.05
        current_step = self.current_step

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            """渲染智能体"""
            agent = rendering.make_circle(unit_x * agent_radius)
            self.agent_trans = rendering.Transform(translation=(unit_x * (1 + self.agent.x), unit_y * (1 + self.agent.y)))  # noqa
            agent.add_attr(self.agent_trans)
            agent.set_color(0.1, 0.3, 0.9)
            self.viewer.add_geom(agent)

            """
            渲染两个障碍物
            """
            # for obstacle in self.obstacles:
            #     rend_obstacle = rendering.make_circle(unit_x * obstacle.r)
            #     self.rend_obstacle_trans = rendering.Transform(translation=(unit_x * (1 + obstacle.x), unit_y * (1 + obstacle.y)))
            #     print(self.rend_obstacle_trans)
            #     rend_obstacle.add_attr(self.rend_obstacle_trans)
            #     rend_obstacle.set_color(0.1, 0.2 ,0.6)
            #     self.viewer.add_geom(rend_obstacle)

            rend_obstacle1 = rendering.make_circle(unit_x * self.obstacles[0].r)
            self.rend_obstacle_trans1 = rendering.Transform(
                translation=(unit_x * self.obstacles[0].x, unit_y * self.obstacles[0].y))
            rend_obstacle1.add_attr(self.rend_obstacle_trans1)
            rend_obstacle1.set_color(0.1, 0.2, 0.8)
            self.viewer.add_geom(rend_obstacle1)

            rend_obstacle2 = rendering.make_circle(unit_x * self.obstacles[1].r)
            self.rend_obstacle_trans2 = rendering.Transform(
                translation=(unit_x * (1 + self.obstacles[1].x), unit_y * (1 + self.obstacles[1].y)))
            rend_obstacle2.add_attr(self.rend_obstacle_trans2)
            rend_obstacle2.set_color(0.1, 0.2, 0.1)
            self.viewer.add_geom(rend_obstacle2)

            t, r, m = 0.1 * unit_x, 0.04 * unit_y, 0.06 * unit_x
            arrow = rendering.FilledPolygon([(t, 0), (m, r), (m, -r)])
            self.arrow_trans = rendering.Transform(rotation=self.agent.theta)  # noqa
            arrow.add_attr(self.arrow_trans)
            arrow.add_attr(self.agent_trans)
            arrow.set_color(0, 0, 0)
            self.viewer.add_geom(arrow)
            """渲染目标"""
            target = rendering.make_circle(unit_x * self.target_radius)
            target_trans = rendering.Transform(translation=(unit_x * 1.5, unit_y * 1.5))
            target.add_attr(target_trans)
            target.set_color(1, 0.5, 0.5)
            self.viewer.add_geom(target)

            mid_target_1 = rendering.make_circle(unit_x * self.mid_target_1_radius)
            mid_target_1_trans = rendering.Transform(translation=(unit_x * (1 + self.mid_target_1.x),1 + unit_y * (1 + self.mid_target_1.y)))
            mid_target_1.add_attr(mid_target_1_trans)
            mid_target_1.set_color(1, 0, 0)
            self.viewer.add_geom(mid_target_1)
        else:
            # self.viewer.geoms = []
            for geom in self.viewer.geoms:
                if isinstance(geom, DrawText):
                    geom.label.delete()
                    self.viewer.geoms.remove(geom)
        text = 'Step: {}'.format(current_step)
        label = pyglet.text.Label(text, font_size=12,
                                  x=10, y=10, anchor_x='left', anchor_y='bottom',
                                  color=(255, 123, 255, 255))
        self.viewer.add_geom(DrawText(label))
        self.arrow_trans.set_rotation(self.agent.theta)
        self.agent_trans.set_translation(unit_x * (1 + self.agent.x), unit_y * (1 + self.agent.y))


        """
        更新障碍物
        """
        # for i in range(self.n_obstacles):
        #     self.rend_obstacle_trans.set_translation(unit_x * (1 + self.obstacles[i].x), unit_y * (1 + self.obstacles[i].y))
        self.rend_obstacle_trans1.set_translation(unit_x * self.obstacles[0].x, unit_y * self.obstacles[0].y)
        self.rend_obstacle_trans2.set_translation(unit_x * (1 + self.obstacles[1].x), unit_y * (1 + self.obstacles[1].y))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# Removed stray duplicate constructor blocks that caused recursion

class MovingEnv(BaseEnv):
    def __init__(
            self,
            seed: int = None,
            max_turn: float = np.pi/2,
            max_acceleration: float = 0.5,
            delta_t: float = 0.005,
            max_step: int = 300,
            penalty: float = 0.001,
            break_value: float = 0.1,
            # 添加三维环境特有的参数
            field_size: float = 1.0,  # 添加三维场大小
    ):
        # 调用父类初始化，但不要传递 field_size（BaseEnv 不接受）
        super(MovingEnv, self).__init__(
            seed=seed,
            max_turn=max_turn,
            max_acceleration=max_acceleration,
            delta_t=delta_t,
            max_step=max_step,
            penalty=penalty,
            break_value=break_value,
        )

        # 在子类中覆盖/设置三维场大小
        self.field_size = field_size

        # 确保使用三维智能体
        self.agent = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )


class SlidingEnv(BaseEnv):
    def __init__(
            self,
            seed: int = None,
            max_turn: float = np.pi/2,
            max_acceleration: float = 0.5,
            delta_t: float = 0.005,
            max_step: int = 300,
            penalty: float = 0.001,
            break_value: float = 0.1,
            # 添加三维环境特有的参数
            field_size: float = 1.0,  # 添加三维场大小
    ):
        # 调用父类初始化，但不要传递 field_size（BaseEnv 不接受）
        super(SlidingEnv, self).__init__(
            seed=seed,
            max_turn=max_turn,
            max_acceleration=max_acceleration,
            delta_t=delta_t,
            max_step=max_step,
            penalty=penalty,
            break_value=break_value,
        )

        # 在子类中覆盖/设置三维场大小
        self.field_size = field_size

        # 确保使用三维智能体
        self.agent = SlidingAgent(
            break_value=break_value,
            delta_t=delta_t
        )