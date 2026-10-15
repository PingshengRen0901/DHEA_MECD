import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加这行导入
import math  # 添加这行导入


class Encoder3D_FeatureEnhanced(nn.Module):
    """特征增强自注意力编码器 (显式QKV实现)"""

    def __init__(self, input_dim=9, hidden_dim=256, output_dim=128):
        super().__init__()

        # 1. 专用子编码器 - 按照9维状态划分（全部复用空间编码器）
        self.agent_encoder = SpatialEncoder3D(input_dim=3, hidden_dim=128, output_dim=64)
        self.target_encoder = SpatialEncoder3D(input_dim=3, hidden_dim=128, output_dim=64)
        self.context_encoder = SpatialEncoder3D(input_dim=3, hidden_dim=96, output_dim=48)

        # 2. 投影层 - 增加维度提升表达能力
        self.d_model = 64
        self.proj_agent = nn.Linear(64, self.d_model)
        self.proj_target = nn.Linear(64, self.d_model)
        self.proj_context = nn.Linear(48, self.d_model)

        # 3. 显式QKV自注意力模块
        self.num_heads = 4
        self.head_dim = self.d_model // self.num_heads

        # Q, K, V投影矩阵
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)

        # 输出投影
        self.W_o = nn.Linear(self.d_model, self.d_model)

        # 特征增强模块 - 更深的网络结构
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(self.d_model * 2, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )

        # 4. 最终融合层 - 更深的网络结构
        self.fusion = nn.Sequential(
            nn.Linear(3 * self.d_model, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 投影层初始化
        for proj in [
            self.proj_agent,
            self.proj_target,
            self.proj_context,
        ]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        # QKV投影初始化
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # 融合层初始化
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # 特征增强模块初始化
        for layer in self.feature_enhancer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def scaled_dot_product_attention(self, Q, K, V):
        """计算缩放点积注意力"""
        # Q, K, V: [batch_size, num_heads, seq_len, head_dim]

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

    def forward(self, x):
        # 1. 特征分组提取
        agent_input = x[:, :3]        # 智能体位置 (x, y, z)
        target_input = x[:, 3:6]      # 目标位置 (x, y, z)
        context_input = x[:, 6:9]     # [距离, 是否到达, 步数比例]

        # 2. 专用子编码器处理
        agent_features = self.agent_encoder(agent_input)
        target_features = self.target_encoder(target_input)
        context_features = self.context_encoder(context_input)

        # 3. 投影到统一维度
        tokens = torch.stack([
            self.proj_agent(agent_features),
            self.proj_target(target_features),
            self.proj_context(context_features),
        ], dim=1)  # [B, 3, d_model]

        # 4. 显式QKV自注意力
        # 生成Q, K, V
        Q = self.W_q(tokens)  # [B, 4, d_model]
        K = self.W_k(tokens)  # [B, 4, d_model]
        V = self.W_v(tokens)  # [B, 4, d_model]

        # 多头分割
        batch_size, seq_len, _ = tokens.size()

        # 重塑为多头格式: [B, seq_len, num_heads, head_dim] -> [B, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 输出投影
        attn_out = self.W_o(attn_output)

        # 5. 特征增强 - 添加残差连接
        enhanced = self.feature_enhancer(attn_out) + attn_out

        # 6. 展平并融合输出
        flattened = enhanced.view(enhanced.size(0), -1)  # [B, 3*d_model]
        return self.fusion(flattened)


class SpatialEncoder3D(nn.Module):
    """三维空间特征专用编码器：处理位置、目标、距离信息"""

    def __init__(self, input_dim=9, hidden_dim=128, output_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.gelu(self.fc1(x))  # 使用GELU激活函数
        return F.gelu(self.fc2(x))  # [B, output_dim]


class DynamicEncoder3D(nn.Module):
    """三维动态特征专用编码器：处理速度、角度、时间信息"""

    def __init__(self, input_dim=6, hidden_dim=64, output_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        return F.gelu(self.fc2(x))  # [B, output_dim]


class EnvironmentEncoder3D(nn.Module):
    """三维环境特征专用编码器：处理洋流、障碍物信息"""

    def __init__(self, input_dim=8, hidden_dim=96, output_dim=48):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        return F.gelu(self.fc2(x))  # [B, output_dim]


# class NoiseEncoder3D(nn.Module):
#     """噪声特征专用编码器：处理噪声信息"""

#     def __init__(self, input_dim=13, hidden_dim=24, output_dim=12):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self._initialize_weights()

#     def _initialize_weights(self):
#         nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
#         nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
#         nn.init.zeros_(self.fc1.bias)
#         nn.init.zeros_(self.fc2.bias)

#     def forward(self, x):
#         x = F.gelu(self.fc1(x))
#         return F.gelu(self.fc2(x))  # [B, output_dim]