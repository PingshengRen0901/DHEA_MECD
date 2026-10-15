import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import wandb
import time
from collections import deque
import random
from PER import *
from hyenv.environment import PITCH
from encoder import *
# 超参数设置
USE_WANDB = False
RENDER = False
RECORD = True
N_STATES = 9  # 状态空间的维度（与环境观测25维对齐）
N_PARAM = 3  # 连续参数的维度
N_ACTIONS = 4  # 动作空间的维度
MEMORY_CAPCITY = 200000  # 记忆库容量
LR_ALPHA = 0.001  # Q网络的学习率
LR_BETA = 0.001  # X网络的学习率
TARGET_PREDICT_ITER = 200  # 目标网络更新频率
BATCH_SIZE = 64  # 每次训练的批次大小
GAMMA = 0.95  # 折扣因子
TOTAL_EPISODE = 25000  # 总训练回合数
MAX_STEP = 300  # 每回合的最大步数
device = "cpu"  # 设备选择


# from encoder import *


class ExpertNetwork(nn.Module):
    """独立专家网络基类"""

    def __init__(self, in_dim, a_dim):
        super(ExpertNetwork, self).__init__()
        # GRU层处理时间序列输入
        self.gru = nn.GRU(in_dim, 256, batch_first=True)
        # 全连接层
        self.encoder = Encoder3D_FeatureEnhanced(
            input_dim=in_dim,
            output_dim=256
        )
        self.fc1 = nn.Linear(256, 64)
        # self.fc2 = nn.Linear(128, 64)
        # 输出层
        self.out = nn.Linear(64, a_dim)
        # 供子类使用的可用组件与属性（提供安全默认值）
        # 加速偏置（默认不偏置）
        self.acceleration_bias = 0.0
        # 额外感知模块：将标量特征映射到动作维度，默认初始化为零以不产生影响
        self.altitude_aware = nn.Linear(1, a_dim)
        self.depth_aware = nn.Linear(1, a_dim)
        self.vertical_distance_aware = nn.Linear(1, a_dim)
        self.vertical_speed_aware = nn.Linear(1, a_dim)
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 使用Xavier初始化替代原来的normal初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.out.weight)
        # 初始化偏置
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.out.bias, 0)
        # 将感知模块初始化为零，不影响初始输出
        for layer in [
            self.altitude_aware,
            self.depth_aware,
            self.vertical_distance_aware,
            self.vertical_speed_aware,
        ]:
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # 添加时间维度
        x = x.unsqueeze(1)
        # GRU处理
        _, x = self.gru(x)
        # 获取最后一个时间步的输出
        x = x[-1, :, :]
        # 全连接层
        x = F.relu(self.fc1(x))

        # 输出Q值
        q = self.out(x)
        return q


class AccelerationExpert(ExpertNetwork):
    """加速专家 - 专门处理加速任务"""
    
    def __init__(self, in_dim, a_dim):
        super(AccelerationExpert, self).__init__(in_dim, a_dim)
        self.acceleration_bias = 0.3
        
    def forward(self, x):
        # 基类前向传播
        q_values = super().forward(x)
        
        # 简单感知：距离远时更倾向于加速
        distance = x[:, 6:7]  # 距离特征
        distance_effect = self.altitude_aware(distance)
        q_values += distance_effect * 0.1
        
        # 对加速动作给予额外奖励
        q_values[:, 0] += self.acceleration_bias
        
        return q_values


class CruisingExpert(ExpertNetwork):
    """巡航专家 - 专门处理稳定巡航任务"""
    
    def __init__(self, in_dim, a_dim):
        super(CruisingExpert, self).__init__(in_dim, a_dim)
        self.cruising_bias = 0.2
        
    def forward(self, x):
        # 基类前向传播
        q_values = super().forward(x)
        
        # 简单感知：中等进度时适合巡航（用step_ratio近似）
        progress = x[:, 8:9]  # 归一化进度
        progress_effect = self.depth_aware(progress - 0.5)
        q_values += progress_effect * 0.1
        
        # 对所有动作给予巡航奖励
        q_values += self.cruising_bias
        
        return q_values


class BrakingExpert(ExpertNetwork):
    """刹车专家 - 专门处理减速任务"""
    
    def __init__(self, in_dim, a_dim):
        super(BrakingExpert, self).__init__(in_dim, a_dim)
        self.braking_bias = 0.4
        
    def forward(self, x):
        # 基类前向传播
        q_values = super().forward(x)
        
        # 简单感知：距离近时更倾向于刹车
        distance = x[:, 6:7]  # 距离特征
        distance_effect = self.altitude_aware(distance)
        q_values += distance_effect * -0.1  # 负权重，距离近时激活
        
        # 对刹车动作给予额外奖励
        q_values[:, 3] += self.braking_bias
        
        return q_values


class HeightAdjustmentExpert(ExpertNetwork):
    """高度调整专家 - 专门处理高度调整任务"""

    def __init__(self, in_dim, a_dim):
        super(HeightAdjustmentExpert, self).__init__(in_dim, a_dim)
        self.height_adjustment_bias = 0.5
        
    def forward(self, x):
        # 基类前向传播
        q_values = super().forward(x)

        # 简单感知：高度误差大时需要调整
        agent_z = x[:, 2:3]  # 智能体高度
        target_z = x[:, 5:6]  # 目标高度
        height_error = agent_z - target_z
        height_error_effect = self.altitude_aware(height_error)
        q_values += height_error_effect * 0.2

        # 对俯仰动作给予额外奖励
        q_values[:, PITCH] += self.height_adjustment_bias

        return q_values

class FineTuningExpert(ExpertNetwork):
    
    """微调专家 - 用于精确控制"""

    def __init__(self, in_dim, a_dim):
        super(FineTuningExpert, self).__init__(in_dim, a_dim)
        self.fine_tuning_bias = 0.3
        
    def forward(self, x):
        # 基类前向传播
        q_values = super().forward(x)

        # 简单感知：进度高时进行微调
        progress = x[:, 8:9]  # 进度比例
        progress_effect = self.vertical_distance_aware(progress)
        q_values += progress_effect * 0.1

        # 对所有动作给予微调奖励
        q_values += self.fine_tuning_bias

        return q_values
class ImprovedMoEGatingNetwork(nn.Module):
    """改进的MoE门控网络 - 决定专家权重"""

    def __init__(self, input_dim, num_experts, hidden_dim=128, k=2):
        """
        Args:
            input_dim: 输入维度
            num_experts: 专家数量
            hidden_dim: 隐藏层维度
            k: 激活的专家数量
        """
        super(ImprovedMoEGatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.k = k

        # 更深的门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )

        # 温度参数控制softmax的锐度
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 计算门控分数
        gate_scores = self.gate(x) / self.temperature

        # Top-k选择
        top_k_values, top_k_indices = gate_scores.topk(self.k, dim=1)

        # 创建稀疏掩码
        mask = torch.zeros_like(gate_scores)
        mask.scatter_(1, top_k_indices, 1)

        # 应用softmax到选中的专家
        sparse_scores = gate_scores * mask
        sparse_scores = sparse_scores - sparse_scores.max(dim=1, keepdim=True).values
        exp_scores = torch.exp(sparse_scores)
        exp_scores = exp_scores * mask  # 确保未选中的专家权重为0
        weights = exp_scores / (exp_scores.sum(dim=1, keepdim=True) + 1e-6)

        return weights


class SumLoss(nn.Module):
    def __init__(self):
        super(SumLoss, self).__init__()

    def forward(self, q_all):
        sigma_q = torch.sum(q_all, 1)
        loss_x = -torch.mean(sigma_q)
        return loss_x


class MoEQNet(nn.Module):
    """MoE Q网络系统"""

    def __init__(self, state_dim, action_dim, num_experts=5, k=2):
        super(MoEQNet, self).__init__()
        self.state_dim = state_dim
        self.num_experts = num_experts
        self.k = k

        # 创建独立专家
        self.experts = nn.ModuleList([
            AccelerationExpert(state_dim, action_dim),
            CruisingExpert(state_dim, action_dim),
            BrakingExpert(state_dim, action_dim),
            HeightAdjustmentExpert(state_dim, action_dim),
            FineTuningExpert(state_dim, action_dim)
        ])

        # 使用改进的门控网络
        self.gating_network = ImprovedMoEGatingNetwork(state_dim, num_experts, k=k)

    def forward(self, x):
        """前向传播方法"""
        # 获取门控权重
        gates = self.gating_network(x)

        # 计算各专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # 加权组合专家输出
        # 扩展维度以进行广播
        gates = gates.unsqueeze(2)
        # 加权求和
        combined_output = torch.sum(expert_outputs * gates, dim=1)

        # 返回组合Q值和门控权重
        return combined_output, gates.squeeze(2)

        # Xnet类，定义了参数网络
class Xnet(nn.Module):
    def __init__(self, s_dim, param_dim):
        super(Xnet, self).__init__()
        self.gru = nn.GRU(s_dim, 256, batch_first=True)
        self.fc1 = nn.Linear(256, 128)  # 替换线性层
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, param_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out.weight)
        # 初始化偏置
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)
        _, x = self.gru(x)
        x = x[-1, :, :]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        param = torch.tanh(self.out(x))  # 使用tanh激活函数来约束输出范围
        return param


class PDQN:
    def __init__(self):
        # 初始化X网络
        self.xnet = Xnet(N_STATES, N_PARAM).to(device)

        # 使用MoE Q网络系统
        self.eval_qnet = MoEQNet(N_STATES + N_PARAM, N_ACTIONS, k=2).to(device)
        self.target_qnet = MoEQNet(N_STATES + N_PARAM, N_ACTIONS, k=2).to(device)

        # 学习计数器
        self.learn_step_counter = 0
        self.memory_counter = 0

        # 经验回放
        self.prioritized = False
        if self.prioritized:
            self.memory = PrioritizedExperienceReplay(capacity=MEMORY_CAPCITY)
        else:
            self.memory = np.zeros((MEMORY_CAPCITY, N_STATES * 2 + 2 + N_PARAM))

        # 优化器
        self.optimizer_q = torch.optim.Adam(self.eval_qnet.parameters(), lr=LR_ALPHA)
        self.optimizer_x = torch.optim.Adam(self.xnet.parameters(), lr=LR_BETA)

        # 专家使用历史
        self.expert_usage_history = []
        self.phase_history = []

    # 在choose_action方法中添加阶段识别
    def choose_action(self, s, episode):
        # 识别当前阶段
        current_phase = self.rule_based_phase_identification(s)
        self.phase_history.append(current_phase)

        s = torch.unsqueeze(torch.FloatTensor(s), 0).to(device)
        params = self.xnet(s)
        input_tensor = torch.cat([s, params], 1)

        # 使用MoE系统获取Q值和门控权重
        actions_value, gate_weights = self.eval_qnet(input_tensor)

        # 记录专家使用情况
        gate_weights_np = gate_weights.detach().cpu().numpy().flatten()
        self.expert_usage_history.append(gate_weights_np)

        params = params.data.cpu().numpy()
        if np.random.uniform() > max(1 - episode * 0.0002, 0.1):
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        else:
            action = np.random.randint(0, N_ACTIONS)
            action = np.array([action])
            params = np.random.uniform(low=[0, -1, -1], high=[1, 1, 1], size=(1, 3))

        return action, params

    def store_transition(self, s, a, x, r, s_):
        x = torch.squeeze(torch.FloatTensor(x), 0)
        transition = np.hstack((s, a, x, [r], s_))
        if self.prioritized:
            self.memory.store(transition)
        else:
            index = self.memory_counter % MEMORY_CAPCITY
            self.memory[index, :] = transition
            self.memory_counter += 1

    def learn(self):
        # 目标网络更新
        if self.learn_step_counter % TARGET_PREDICT_ITER == 0:
            self.target_qnet.load_state_dict(self.eval_qnet.state_dict())
        self.learn_step_counter += 1

        # 从经验池中采样
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(BATCH_SIZE)
            ISWeights = torch.FloatTensor(ISWeights).to(device).view(BATCH_SIZE, 1)
        else:
            sample_index = np.random.choice(MEMORY_CAPCITY, BATCH_SIZE)
            batch_memory = self.memory[sample_index, :]
            ISWeights = torch.ones(BATCH_SIZE, 1).to(device)

        # 提取批处理数据
        b_s = torch.FloatTensor(batch_memory[:, :N_STATES]).to(device)
        b_a = torch.LongTensor(batch_memory[:, N_STATES:N_STATES + 1].astype(int)).to(device)
        b_x = torch.FloatTensor(batch_memory[:, N_STATES + 1:N_STATES + 1 + N_PARAM]).to(device)
        b_r = torch.FloatTensor(batch_memory[:, N_STATES + 1 + N_PARAM:N_STATES + 1 + N_PARAM + 1]).to(device)
        b_s_ = torch.FloatTensor(batch_memory[:, -N_STATES:]).to(device)

        # 计算Q网络损失
        q_eval, _ = self.eval_qnet(torch.cat((b_s, b_x), 1))
        q_eval = q_eval.gather(1, b_a)

        next_params = self.xnet(b_s_)
        q_next, _ = self.target_qnet(torch.cat((b_s_, next_params.detach()), 1))
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        if self.prioritized:
            abs_errors = torch.abs(q_target - q_eval).detach().cpu().numpy()
            self.memory.batch_update(tree_idx, abs_errors)
            loss_q = (ISWeights * F.mse_loss(q_eval, q_target, reduction='none')).mean()
        else:
            loss_q = F.mse_loss(q_eval, q_target)

        # 清零梯度并反向传播Q网络损失
        self.optimizer_q.zero_grad()
        loss_q.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.eval_qnet.parameters(), max_norm=1.0)
        self.optimizer_q.step()

        # 计算参数网络损失（使用新的前向传播）
        params = self.xnet(b_s)
        actions_value, _ = self.eval_qnet(torch.cat((b_s, params), 1))
        loss_x = -torch.mean(actions_value)

        # 清零梯度并反向传播X网络损失
        self.optimizer_x.zero_grad()
        loss_x.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.xnet.parameters(), max_norm=1.0)
        self.optimizer_x.step()

        return loss_q.item(), loss_x.item()

、
