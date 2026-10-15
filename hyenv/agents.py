from itertools import product

import numpy as np

from itertools import product
import numpy as np

class BaseAgent:
    def __init__(self, break_value: float, delta_t: float):
        self.x = None
        self.y = None
        self.z = None  # 添加z坐标
        self.theta = None  # 偏航角（水平方向）
        self.phi = None    # 俯仰角（垂直方向）
        self.speed = None
        self.delta_t = delta_t
        self.break_value = break_value

    def accelerate(self, value: float) -> None:
        raise NotImplementedError

    def break_(self) -> None:
        raise NotImplementedError

    def turn(self, value: float) -> None:
        raise NotImplementedError

    def pitch(self, value: float) -> None:
        raise NotImplementedError

    def reset(self, x: float, y: float, z: float, theta: float, phi: float) -> None:
        """重置智能体位置和方向（三维）"""
        self.x = x
        self.y = y
        self.z = z
        self.speed = 0
        self.theta = theta  # 偏航角
        self.phi = phi      # 俯仰角

    def _step(self) -> None:
        """更新位置（三维）"""
        # 计算速度分量
        vx = self.speed * np.cos(self.phi) * np.cos(self.theta)
        vy = self.speed * np.cos(self.phi) * np.sin(self.theta)
        vz = self.speed * np.sin(self.phi)
        
        # 更新位置
        self.x += self.delta_t * vx
        self.y += self.delta_t * vy
        self.z += self.delta_t * vz


class MovingAgent(BaseAgent):
    def __init__(self, break_value: float, delta_t: float):
        super(MovingAgent, self).__init__(break_value, delta_t)

    def accelerate(self, value: float) -> None:
        """加速（三维）"""
        self.speed += value
        self._step()

    def break_(self) -> None:
        """刹车（三维）"""
        self.speed = 0 if self.speed < self.break_value else self.speed - self.break_value
        self._step()

    def turn(self, value: float) -> None:
        """水平转向（改变偏航角）"""
        self.theta = (self.theta + value) % (2 * np.pi)
        self._step()
    
    def pitch(self, value: float) -> None:
        """俯仰控制（改变俯仰角）"""
        new_phi = self.phi + value
        self.phi = max(min(new_phi, np.pi/12), -np.pi/12)
        self._step()


class SlidingAgent(BaseAgent):
    def __init__(self, break_value: float, delta_t: float):
        super(SlidingAgent, self).__init__(break_value, delta_t)
        # 在滑动模型中，速度方向可能与朝向不同
        self.velocity_theta = 0  # 速度的偏航角
        self.velocity_phi = 0    # 速度的俯仰角

    def accelerate(self, value: float) -> None:
        """加速（三维滑动模型）"""
        # 计算当前速度分量
        vx_current = self.speed * np.cos(self.velocity_phi) * np.cos(self.velocity_theta)
        vy_current = self.speed * np.cos(self.velocity_phi) * np.sin(self.velocity_theta)
        vz_current = self.speed * np.sin(self.velocity_phi)
        
        # 加速分量（沿当前朝向）
        ax = value * np.cos(self.phi) * np.cos(self.theta)
        ay = value * np.cos(self.phi) * np.sin(self.theta)
        az = value * np.sin(self.phi)
        
        # 合成新速度
        vx_new = vx_current + ax
        vy_new = vy_current + ay
        vz_new = vz_current + az
        
        # 计算新速度大小和方向
        self.speed = np.sqrt(vx_new**2 + vy_new**2 + vz_new**2)
        self.velocity_theta = np.arctan2(vy_new, vx_new)
        self.velocity_phi = np.arctan2(vz_new, np.sqrt(vx_new**2 + vy_new**2))
        
        self._step()

    def break_(self) -> None:
        """刹车（三维滑动模型）"""
        self.speed = 0 if self.speed < self.break_value else self.speed - self.break_value
        # 如果速度为零，速度方向重置为当前朝向
        if self.speed == 0:
            self.velocity_theta = self.theta
            self.velocity_phi = self.phi
        self._step()

    def turn(self, value: float) -> None:
        """水平转向（改变偏航角）"""
        self.theta = (self.theta + value) % (2 * np.pi)
        self._step()
    
    def pitch(self, value: float) -> None:
        """俯仰控制（改变俯仰角）"""
        new_phi = self.phi + value
        self.phi = max(min(new_phi, np.pi/120), -np.pi/120)
        self._step()
    
    def _step(self) -> None:
        """更新位置（三维滑动模型）"""
        # 使用速度方向而不是朝向
        vx = self.speed * np.cos(self.velocity_phi) * np.cos(self.velocity_theta)
        vy = self.speed * np.cos(self.velocity_phi) * np.sin(self.velocity_theta)
        vz = self.speed * np.sin(self.velocity_phi)
        
        self.x += self.delta_t * vx
        self.y += self.delta_t * vy
        self.z += self.delta_t * vz


class HardMoveAgent(BaseAgent):
    def __init__(self, break_value: float, delta_t: float, num_actuators: int = 6):
        super(HardMoveAgent, self).__init__(break_value, delta_t)
        self.num_actuators = num_actuators
        self.K = 2 ** self.num_actuators
        self.meta_to_mask = list(product(*[list(range(2)) for _ in range(self.num_actuators)]))
        self.move_vector = np.zeros(3)  # 三维移动向量

    def accelerate(self, value: float) -> None:
        """硬移动模型不使用加速方法"""
        pass

    def break_(self) -> None:
        """硬移动模型不使用刹车方法"""
        pass

    def turn(self, value: float) -> None:
        """硬移动模型不使用转向方法"""
        pass

    def pitch(self, value: float) -> None:
        """硬移动模型不使用俯仰方法"""
        pass

    def move(self, move_direction_meta: int, move_distances: list) -> None:
        """三维硬移动模型"""
        move_directions_mask = self.meta_to_mask[int(move_direction_meta)]
        
        # 三维方向向量
        directions = [
            np.array([np.cos(i * 2 * np.pi / self.num_actuators), 
                      np.sin(i * 2 * np.pi / self.num_actuators),
                      0]) for i in range(self.num_actuators)
        ]
        
        # 添加垂直方向
        directions.append(np.array([0, 0, 1]))  # 向上
        directions.append(np.array([0, 0, -1])) # 向下
        
        # 计算移动向量
        self.move_vector = np.zeros(3)
        for i in range(min(len(move_distances), len(directions))):
            if move_directions_mask[i]:
                self.move_vector += move_distances[i] * directions[i]
        
        self._step()
        # 更新朝向（仅水平方向）
        self.theta = np.arctan2(self.move_vector[1], self.move_vector[0])

    def _step(self) -> None:
        """更新位置（三维硬移动模型）"""
        self.x += self.move_vector[0]
        self.y += self.move_vector[1]
        self.z += self.move_vector[2]
