# DHEA-MECD：论文方法与实验代码

本仓库实现论文 **Efficient Mobile Intelligence for AUV-Based Multi-Target Tracking via Structured Perception and Multi-Expert Reinforcement Learning** 的核心方法，并提供八种算法对比、观测维度实验、模块消融、Top-k 扫描、三种子运行、评估、统计和绘图流程。

**状态说明：代码和短程功能验证已经完成；完整 10,000 回合实验及论文数值尚未复现。** 四组维度已逐项核对作者本次提供的 `1015appendix.pdf`（2 页），特征身份与索引顺序一致，并将该文件登记为正式依据。来源指纹保存在 `experiments/sources.json`，核对记录见 `docs/appendix_alignment.md`。论文及附录没有完整指定的公式、环境、基线适配和超参数，均在下文明确列出。

## 1. 安装与快速开始

要求 Python 3.10+。训练不需要图形桌面、wandb 或旧版 Gym。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

# 使用 Table I 的训练默认值和 36D 观测
python train.py --config experiments/paper.json --episodes 10000 --output runs/dhea36_seed42

# 使用命令行指定另一组观测和算法
python train.py --algorithm ppo --dimension 18 --seed 43 --output runs/ppo18_seed43
```

请在已安装 PyTorch 的虚拟环境中运行上述命令，并确保 `python` 与安装依赖时使用的解释器一致。

先做短程验证：

```bash
# 所有算法、维度、模块消融、k 扫描，共 40 个单种子配置
python run_experiments.py --suite all --seeds 42 --smoke --output runs/smoke
python summarize.py runs/smoke
python plot_results.py runs/smoke/summary

# 三种子统计流程的小规模验证
python run_experiments.py --suite comparison --algorithms dhea_mecd --dimensions 36 \
  --seeds 42 43 44 --smoke --output runs/smoke_three_seeds
```

`--smoke` 每个配置训练 1 回合、每回合 8 步、评估 2 回合，并缩小网络和 batch。配置、统计与图标题均标记为 smoke，不可作为论文实验结果。正常实验不要添加此选项。

## 2. 文件结构

| 文件 | 用途 |
| --- | --- |
| `config.py` | 观测语义布局、网络和学习参数 |
| `encoder.py` | 四分支编码、六头注意力、信息增强、无结构编码消融 |
| `pdqn.py` | DHEA-MECD、普通 PDQN 单 Q 头、目标网络、PER 学习与检查点 |
| `PER.py` | SumTree、重要性权重、因果历史重建、环形边界处理 |
| `hyenv/agents.py` | 三维粒子动力学、相对流速阻力与混合动作执行 |
| `hyenv/disturbances.py` | RBF 洋流与四类水下噪声 |
| `hyenv/observations.py` | appendix 四组维度的逐项特征与归一化 |
| `hyenv/environment.py` | 多目标交互、障碍物、奖励、终止、轨迹和能耗 |
| `algorithms/baselines.py` | PPO、SAC、DDPG、TRPO、AC、DQN |
| `train.py` | 所有算法统一训练、续训、评估和时延记录 |
| `run_experiments.py` | 完整实验矩阵、配置指纹、运行日志与中断恢复 |
| `summarize.py` | 以训练种子为独立单位统计、CSV、95% CI、能耗节省 |
| `plot_results.py` | 从实际 CSV 绘制可编辑 SVG/PDF 与 600 dpi PNG |
| `visualize_trajectory.py` | 读取评估轨迹，使用 PyVista 输出三维可视化 |
| `experiments/paper.json` | Table I 默认值与全部显式实现选择 |
| `experiments/features.json` | 9/18/27/36D 特征顺序清单 |
| `tests/` | 算法、环境、观测、统计及检查点回归测试 |

## 3. 方法与论文的对应

| 论文位置 | 实现 | 关键行为 |
| --- | --- | --- |
| III-A，式 (1)–(4) | `OceanCurrent`、`MovingAgent` | 时变 RBF 流场、惯性速度减局部洋流、阻力与外力积分 |
| III-B，式 (5)–(11) | `UnderwaterNoise` | 随速度变化的高斯噪声、对称 α 稳定噪声、低频有色噪声、相关湍流噪声 |
| V-A，式 (12)–(17) | `encoder.py` | space：GELU/残差/LayerNorm；motion：Swish/GroupNorm；barrier：ReLU/残差/LayerNorm；noise：ReLU/BatchNorm |
| V-A，式 (18)–(23) | `Encoder3D_FeatureEnhanced` | 模态投影、六头自注意力、残差增强、扁平化策略表征 |
| V-B，式 (24)–(25) | `Xnet` | 历史状态 GRU、两层 ReLU MLP、四维 tanh 输出 |
| V-B，式 (26)–(27) | `ImprovedMoEGatingNetwork` | 潜在状态与可学习专家键双线性匹配、Top-k 稀疏 softmax |
| V-B，式 (28)–(30) | `ExpertNetwork`、`MoEQNet` | 五个独立 GRU-MLP、可学习阶段感知、稀疏执行、加权 Q 与离散 argmax |
| Algorithm 1，式 (31)–(36) | `PDQN.learn` | PER、终止屏蔽、目标连续策略、加权 TD、固定 Q 的连续策略更新、目标软更新 |
| VI-A，式 (37)–(41) | `MovingEnv.step` | 距离进展、目标获取、碰撞、边界四项奖励相加 |
| VII-B，式 (42)–(43) | 环境与 `summarize.py` | 推进/转向指令平方积分除以 E_ref；相对 PPO 的能耗节省 |
| Fig. 4–6 / Table II | 实验 comparison / 统计 | 四组维度的学习曲线、成功率、碰撞率、成功回合路径分布 |
| Fig. 7 | 实验 ablation | w/o encoder-attention 与 w/o MECD，18D / 36D |
| Fig. 8 | 实验 topk | 固定专家数 5，k 从 1 到 5，36D |
| Fig. 9–11 | 评估 / 绘图 / PyVista | 推理时延、归一化能耗、三维轨迹 |

未选中的专家不执行该样本的前向计算，也不接收该样本的梯度。不同样本选择不同专家时，一个 batch 可以涉及全部专家，但每个样本仅路由到 k 个专家。

## 4. 四组观测：严格保留维度说明中的索引顺序

| 维度 | 按原始索引排列的内容 | 四分支输入维度（space/motion/barrier/noise） |
| --- | --- | --- |
| 9 | 1–3 局部相对位移；4 距离；5 捕获指示；6–8 速度；9 时间 | 5 / 4 / 0 / 0 |
| 18 | 1–3 AUV 位置；4–6 目标质心；7 距离；8 捕获指示；9–11 速度；12–15 偏航/俯仰 sin/cos；16 时间；17–18 水面/海床距离 | 8 / 8 / 2 / 0 |
| 27 | 1–16 同 18D 的前 16 项；17–19 洋流；20–23 最近障碍物 1 相对 xyz 与半径；24–27 最近障碍物 2 同类信息 | 8 / 8 / 11 / 0 |
| 36 | 1–3 AUV 位置；4–6 目标质心；7–9 相对引导；10 距离；11 捕获指示；12–14 速度；15–18 姿态 sin/cos；19 偏航角速度；20 时间；21–23 洋流；24–31 两障碍物状态；32–35 四类噪声；36 感知置信度 | 11 / 9 / 11 / 5 |

编码器根据显式索引抽取语义组。低维配置不包含的模态以零 token 和注意力 key mask 表示，不从环境额外读取这些信息。所有对比算法获得同一配置下完全相同的归一化观测。

所有特征映射到 `[0,1]`。位置、速度、洋流按固定物理边界缩放；噪声用 arctan 映射以保留重尾顺序；9D 位移通过当前姿态转换到局部坐标系。水面为 `z=+field_size`，海床为 `z=-field_size`。障碍物每步按当前距离排序，不足两个时以中性位移和零半径填充。

附录写明“目标群质心”，但未说明是否排除已获取目标。当前使用尚未获取目标的质心；全部获取后使用全部目标质心。36D 引导向量采用世界坐标系的目标位移，没有额外单位向量化；四类噪声使用正文随机过程的样本值，按 arctan 映射。置信度附录没有公式，当前明确选择 `1 / (1 + mean(abs(noise)) / noise_std)`。这些构造选择不能被当成附录已给出的数学定义。

为兼容完整状态研究，仍支持 `--dimension full`：直接保留所有目标位置、距离和累计捕获标记，总维度 `18 + 5J + 3N`（3 目标、2 障碍物时 39D）。四组论文维度是分别构造的特征集，不是对 39D 简单截断。Python 直接调用 `MovingEnv()` 默认使用完整状态；训练 CLI 默认 36D。

## 5. 混合动作、动力学和多目标任务

动作接口为 `(discrete_id, parameters)`。离散动作固定为 `0=加速、1=偏航、2=俯仰、3=制动`，四维参数为 `[acceleration, yaw_rate, pitch_rate, braking]`，均位于 `[-1,1]`。加速/制动映射到 `[0,1]` 强度，角速度保留符号；只执行当前离散动作对应的参数。

默认使用一个 AUV 和三个动态目标；目标与障碍物匀速运动，在工作空间边界反射。每个目标进入捕获半径后只奖励一次，全部目标捕获才算成功。距离进展针对本步开始前最近的未捕获目标，计算前后始终保持同一个目标身份，避免切换目标产生虚假奖励。碰撞/越界时不给额外捕获奖励。

全部获取、碰撞或越界返回 `terminated=True`；达到时间上限返回 `truncated=True`。时间截断继续自举，任务终止停止自举，历史序列不跨 reset。

采用 Gymnasium API：`reset() -> (state, info)`，`step() -> (state, reward, terminated, truncated, info)`。注册 `Moving-v0` 和 `AUVTracking-v0`。旧滑动模型及旧版 9D/3 参数检查点不属于当前实现。

## 6. 参数来源与无法从论文唯一确定的部分

Table I 中直接采用：10,000 回合、800 步、专家数 5、k=2、专家 MLP 64、两个学习率 0.001、γ=0.95、batch=64、容量 200,000、ε 初值 1.0、每回合衰减 0.0002、梯度裁剪 1.0、Δt=0.005、推进/转向权重均为 1。

以下为显式实现选择，集中记录在 `experiments/paper.json` 和相关类中：

- Table I 写 `T_update=200`，Algorithm 1 / 式 (36) 写每次学习软更新，未说明如何组合。本实现默认遵循 Algorithm 1，τ=0.005；可用 `--target-update hard` 每 200 次学习硬更新。JSON 配置运行时应修改 `agent.target_update`，不能认为两种设定同时等价于论文。
- 编码器隐藏宽度 128、token 96、latent 128、GRU 256、dropout=0.1、历史长度 8、ε 下限 0.1、warmup=1000 均未在 Table I 完整给出。历史采用固定窗口和左侧首帧重复填充；经验库在回合/环形覆盖边界缩短上下文。
- 经验库保存原始观测，在学习时重算 latent，避免缓存表征在编码器更新后过期。编码器随 TD 损失更新；连续策略更新时固定编码器和 Q，并按式 (35) 使用经验中的离散动作。
- 五专家角色标为加速、巡航、制动、高度调整、微调。式 (28) 未定义每个角色的精确特征变换，因此采用独立可学习潜在特征权重和 MLP；角色名不能保证训练后自动形成对应物理行为。
- 动力学为单位惯性矩阵的三维粒子近似，含线性/二次相对流速阻力，不是完整六自由度水动力模型。四类噪声之和限幅后施加到三轴外力；密度固定在参考值，vehicle 噪声标准差按速度线性缩放，即速度方差指数取 2。
- 场地半边长 1、目标半径 0.1、AUV 半径 0.025、障碍物半径 0.08、目标速度 0.05、障碍物速度 0.02、洋流尺度 0.1、噪声尺度 0.01；质量/阻力、RBF 基数、噪声相关系数等也属于未给出的实现选择。可通过环境构造参数或代码内扰动类调整。
- 逐个获取目标、最近目标奖励、未获取目标质心观测、初始分布和边界反射均不是论文完整规定的任务细节。
- `E_ref=1` 为可配置的共同归一化常数。能耗衡量执行的归一化指令成本，不是真实电池能量。缺少物理标定时 `length_scale_m=null`，路径单位标记为 simulation unit，不输出伪造的 200–500 m 表格。提供每仿真单位对应米数后，设置该参数即可按 Table II 区间汇总。
- 推理时延测量实际运行设备，并记录环境/库版本。当前 CPU 验证结果不能与论文 RTX 4090 的毫秒数直接对照。

这份代码不写入论文报告的成功率、能耗、时延或轨迹分布作为训练/测试输出。

## 7. 七种对比基线与公平性

所有基线使用相同四分支编码和六头注意力架构、同一维度观测、同一环境、相同训练回合/步数上限、相同训练种子和留出评估场景。网络分别训练，不跨算法共享已训练权重。各策略后接历史 GRU。

| 算法 | 实际更新 | 混合动作适配 |
| --- | --- | --- |
| PDQN | 单 Q 头、连续策略、Bellman TD 与确定性策略梯度、均匀回放 | 与 DHEA-MECD 同样的四离散/四连续动作 |
| PPO | 行为 log-prob 比率、clipped surrogate、GAE、价值损失 | 四类 categorical + 四维 tanh-Normal，包含 tanh 的 log-Jacobian |
| AC | 普通 advantage policy gradient、价值回归 | 同 PPO 的混合分布，不使用 PPO clipping |
| TRPO | KL Hessian-vector product、共轭梯度、KL 步长、回溯线搜索；不满足约束则回滚 | 同 PPO 的混合分布，KL 对 categorical 与 Normal 求和 |
| DQN | 离散 Q、均匀回放、目标 Q、TD 更新 | 每种动作使用 5 个参数档位，构成 20 个动作 |
| DDPG | 确定性 actor、Q、目标网络、软更新、Gaussian 探索 | 八维连续向量：前四维 argmax 选离散动作，后四维作为参数 |
| SAC | 随机 actor、双 Q、最小目标 Q、熵正则、重参数梯度、软更新 | 同八维连续适配，tanh-Normal，固定 α=0.2 |

论文没有给出上述基线适配规则及完整基线超参数，这些是透明的比较实现，不应表述成作者原始基线代码的逐行复刻。论文 DQN 引文对应 bootstrapped DQN，但正文只称 DQN；这里采用标准单头 DQN，未假定未写明的 bootstrap 头数。

PPO/TRPO/AC 在策略比率与 KL 更新期间保持 dropout 关闭、BatchNorm 运行统计固定，BN 的仿射参数仍学习，以保证同一状态的策略分布可重复计算。其他基线也采用确定性前端。DHEA-MECD 的 TD 编码器训练使用 dropout 与批统计；评估统一关闭 dropout、使用运行统计。

基线实现选择见 JSON：rollout=256、PPO/价值迭代=10、clip=0.2、GAE λ=0.95、entropy coefficient=0.01、value coefficient=0.5、TRPO max KL=0.01、CG=10、damping=0.1、线搜索=10、DDPG noise=0.1、DQN 档位数=5。这些值不是 Table I 已报告的全部参数。

算法原始参考：[PPO](https://arxiv.org/abs/1707.06347)、[TRPO](https://arxiv.org/abs/1502.05477)、[SAC](https://arxiv.org/abs/1812.05905)、[DDPG](https://arxiv.org/abs/1509.02971)。

## 8. 完整实验、消融和 k 扫描

先生成计划，检查配置和预算：

```bash
python run_experiments.py --suite all --plan-only --output runs/paper
```

默认三独立训练种子为 42、43、44。完整矩阵共 **120 个独立训练任务**：96 个八算法/四维度对比、12 个两模块消融、12 个额外 k 配置；完整模型及 k=2 结果复用，不重复训练。

```bash
# 完整实验：每任务 10000 训练回合、100 个留出评估回合
python run_experiments.py --suite all --device cuda --output runs/paper

# 可以分开运行
python run_experiments.py --suite comparison --output runs/comparison
python run_experiments.py --suite ablation --output runs/ablation
python run_experiments.py --suite topk --output runs/topk

# 只运行指定算法与维度
python run_experiments.py --suite comparison --algorithms dhea_mecd pdqn \
  --dimensions 18 36 --seeds 42 43 44 --output runs/subset
```

`ablation` 单独运行包括 18D/36D 全模型参考，共 18 任务；`topk` 单独运行包括 k=2，共 15 任务。`--dimensions` / `--algorithms` 只筛选 comparison 部分，论文规定的消融和 k 维度保持不变。

重新执行同一个 runner 命令会读取 manifest，跳过完成任务，并从未完成任务的最近检查点恢复。配置指纹、训练/评估预算或设备发生变化时要求使用新输出目录。未写入检查点的 CSV 尾部回合会移除并重新执行；中断的评估按固定种子完整重跑。

单任务也可以续训和评估：

```bash
python train.py --resume runs/dhea36_seed42/latest.pt --episodes 100 --output runs/dhea36_seed42
python train.py --evaluate runs/dhea36_seed42/latest.pt --episodes 100 \
  --evaluation-seed 1000000 --save-trajectories --output runs/dhea36_seed42/evaluation
```

`--episodes` 是本次新增回合数。续训自动恢复环境、seed、warmup 和网络配置，CLI 的默认环境参数不会覆盖检查点。检查点按回合边界保存网络、优化器、回放库、训练计数和 Python/NumPy/PyTorch/CUDA/探索/采样 RNG。旧版本缺少环境元信息的检查点需要迁移，不能默认为同一实验。

配置文件中的设置优先于 CLI 中对应的训练/环境/网络默认值。设备、输出目录和本次运行回合数仍由 CLI 控制。训练期间的 checkpoint 默认每 100 回合保存，并在本次最后一回合保存。

## 9. 评估、统计和输出

```bash
python summarize.py runs/paper
python plot_results.py runs/paper/summary
```

`run_experiments.py` 默认使用训练种子 42/43/44、独立评估种子序列 1,000,000…，所有算法共享评估场景。评估关闭探索且不更新经验库或网络；初始 10 次推理为时延预热。每步时延包括历史组装、CPU/设备数据传输、编码和策略选择，排除环境 step；CUDA/MPS 使用同步计时。日志中的 p95 为各回合内部的 p95，汇总字段明确命名为 mean_episode_p95_ms，不冒充合并全部步数的全局 p95。

单任务输出：

- `training_config.json`、`training.csv`、`latest.pt`、运行日志。
- `evaluation/evaluation_config.json`、`evaluation.csv`、`summary.json`。
- 评估轨迹 `evaluation/trajectories/episode_*.npz`，包含 AUV、目标、障碍物路径与 Δt。

汇总输出：

- `per_seed.csv`：每个独立训练种子的评估均值。
- `summary.csv`：种子间均值、样本 SD、双侧 95% Student t CI（df=n−1）。
- `learning_curves.csv`：每个训练回合的跨种子回报及 CI；不平滑、不插值。
- `energy_episode_sd.csv`：按论文能耗段落另报评估回合上的描述性均值与 SD，不把这些回合当独立训练重复。
- `energy_savings_vs_ppo.csv`：相同训练种子下相对 PPO 的能耗节省百分比。
- `trajectory_distribution.csv`：只在有明确米制标定或显式 `--length-bins` 时生成；分母为成功回合，区间外的成功路径也保留，避免隐藏数据。
- `statistics.json`：统计单位、区间定义、协议类型和未完成任务排除记录。
- `figures/`：收敛、成功率、碰撞率、时延、能耗、模块消融和 k 扫描的 SVG/PDF/PNG，以及数据来源说明。

无成功回合时，“成功条件下平均轨迹长度”记为缺失，不填成 0。单种子不绘制 CI。默认不允许汇总未完成的矩阵；如明确使用 `--allow-incomplete`，排除任务会写入统计说明。三种子 CI 的推断能力有限，不代表对未知真实水域的保证。

对于 Table II 的米制区间，须先明确 `environment.length_scale_m`；之后默认统计 200–300、300–350、350–400、400–500 m，并记录区间外数量。没有该物理尺度时不会从仿真长度臆造米数。

## 10. 三维轨迹可视化

```bash
python -m pip install -r requirements-visualization.txt
python visualize_trajectory.py runs/dhea36_seed42/evaluation/trajectories/episode_0.npz \
  --output runs/dhea36_seed42/trajectory.png
```

AUV 使用黄色球和轨迹，目标使用红色球和轨迹，障碍物为半透明灰色。默认离屏截图；`--interactive` 打开交互窗口。PyVista 在独立进程中渲染，运行环境需要支持 VTK/OpenGL。原生渲染失败时自动生成标题明确标注的 Matplotlib 轨迹预览，并在图片同名 `.json` 中记录实际渲染器和失败原因；这不算 PyVista 渲染已通过。`--strict-pyvista` 禁止回退；`--backend matplotlib` 可直接生成预览。

## 11. 已验证与尚待完成

已验证：25 项测试；作者本次附录的四组特征身份与索引顺序；局部坐标、速度/时间、姿态、水面/海床距离、障碍物半径与空槽、偏航角速度、四类噪声和置信度槽位；六类新增基线在四维度下实际更新参数；TRPO KL 约束和可二阶求导 GRU 的数值一致性；低维缺失模态屏蔽；全部模块消融和 k 值；PER/历史边界；目标更新；检查点恢复；完整 120 任务计划；40 个单种子配置的短程训练与留出评估；三种子统计与图形导出；Matplotlib 三维轨迹预览及 PyVista 失败后的进程隔离回退。当前会话无法创建 PyVista 的 OpenGL 上下文，原生 PyVista 截图尚未在本机会话中通过。CUDA/MPS 的实际训练尚未在本机会话中验证。

尚待作者材料或计算完成：确认附录未给出公式的特征构造、未公开的环境标定、基线动作适配与超参数；解决 Table I 和 Algorithm 1 的目标更新说明差异；在正式平台跑完整 10,000 回合 × 三种子实验，才能检验是否达到论文报告的成功率、收敛、能耗和时延。附录版本核对已完成，当前不声称已完成数值复现。
