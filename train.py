"""
    TODO:改成优势函数
    DONE:保存网络的参数，随机种子的设置
    DONE：统计离散动作出现的频率
    DONE:记录图像，wandb


2.23
等到后面输入的数据多了，可以用conv层进行提炼，然后又可以用network in network 用1层conv层代替非线性函数，继续减少数据量
"""
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
import hyenv
from pdqn import *
from PER import SumTree, PrioritizedExperienceReplay
from pdqn import PDQN, SumLoss

"""超参数"""
#训练监控
USE_WANDB = False #True False
RENDER = False
RECORD = False
USE_PER = True
USE_GRU = True
N_STATES = 9
N_PARAM = 3
N_ACTIONS = 4
MEMORY_CAPCITY = 200000#200000  10w 600episode可存完
LR_ALPHA =0.001
LR_BETA = 0.001
TARGET_PREDICT_ITER = 300
BATCH_SIZE = 64
GAMMA = 0.95
TOTAL_EPISODE = 10000 # 训练的epo 25000
MAX_STEP = 300

import os, torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# try:
#     train_env.reset(seed=seed)
# except TypeError:
#     train_env.seed(seed)
"""训练"""
if USE_WANDB:
    wandb.init(project='hybrid', name='9-26-v1') #7defb26fc369c2b79a47db3d342bf51d15d2d597



pdqn = PDQN()
env = gym.make('Moving-v0', max_step=MAX_STEP)
if RECORD:
    # env = gym.wrappers.Monitor(env, "record", video_callable=lambda episode_id: episode_id % 5 == 0 and episode_id > 5000, force=True)
    env = gym.wrappers.Monitor(env, "record", video_callable=lambda episode_id: episode_id % 100 == 0, force=True)

start_time = time.time()
for episode in range(TOTAL_EPISODE):
    if episode % 100 == 0 and episode != 0:
        with open('testing.txt', 'a') as f:
            f.write('\n')  # 换行
            f.write("episode为：{}".format(episode))

    done = False
    s = env.reset()
    episode_reward = 0
    reward_list = []
    step = 0
    cnt_acc = 0
    cnt_turn = 0
    cnt_break = 0
    while not done:  # 200步
        if RENDER:
            env.render()
        step += 1
        a, x = pdqn.choose_action(s, episode)  # s是state episode用来控制探索的概率
        current_state = s
        if a == [0]:
            cnt_acc += 1
        elif a == [1]:
            cnt_turn += 1
        else:
            cnt_break += 1
        # 把输出的a和x转换成一个元组
        x = x[0]
        a = a[0]
        mix_action = (a, x)

        next_state, reward,  done, reward_list = env.step(mix_action)  # 输入离散和连续动作，输出下一步这步的奖励
        episode_reward += reward
        pdqn.store_transition(s, a, x, reward, next_state)
        s = next_state
        # 小数点近似
        decimal_places = 4
        formatted_current_state = [format(_, f".{decimal_places}f") for _ in current_state]
        formatted_next_state = [format(_, f".{decimal_places}f") for _ in next_state]

        # 小数点近似
        if episode % 50 == 0 and episode != 0:
            with open('testing.txt', 'a') as f:
                f.write('\n')
                f.write("step:{}\n当前状态:{}\n下一状态：{}\n离散动作:{}\n连续动作:{}\nstep奖励:{}\n总奖励为：{}\n目前的flag为：{}\nreward_list为：{}\n".format(step,
                                             formatted_current_state, formatted_next_state, a, x, reward, episode_reward, current_state[8], reward_list))
                f.write("--------------------")
        if pdqn.memory_counter > MEMORY_CAPCITY:
            pdqn.learn()


    # 每两次打印一次信息
    if episode % 2 == 0 and episode != 0:
        # 准备打印信息
        episode_info = "当前episode为：{}\n总奖励为：{}\ni为：{}\neps为：{}".format(episode, episode_reward, step, max(1 - episode*0.0002, 0.1))
        action_info = "加速动作共：{},转弯动作共：{},刹车动作共：{}".format(cnt_acc, cnt_turn, cnt_break)
        memory_info = "经验回放已经存储了：{}".format(pdqn.memory_counter) if pdqn.memory_counter <= MEMORY_CAPCITY else "经验回放已满"
        
        # 打印到控制台
        print(episode_info)
        print(action_info)
        print(memory_info)
        print("-------------")
        
        # 写入freward.txt文件
        with open('freward.txt', 'a', encoding='utf-8') as f:
            f.write(f"Episode: {episode}\n")
            f.write(f"总奖励: {episode_reward}\n")
            f.write(f"步数: {step}\n")
            f.write(f"探索率: {max(1 - episode*0.0002, 0.1)}\n")
            f.write(f"加速动作: {cnt_acc}, 转弯动作: {cnt_turn}, 刹车动作: {cnt_break}\n")
            f.write(f"经验回放: {memory_info}\n")
            f.write("--------------------\n")
        
        if USE_WANDB:
            wandb.log({'episode': episode, 'episode_reward': episode_reward, 'eps': max(1 - episode*0.0002, 0.1)})


    # print("episode:{},step:{}".format(episode, step))
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        if episode % 100 == 0 and episode > 0:
            # 收集随机数状态，便于完全无缝续训
            torch_rng_state = torch.get_rng_state()
            cuda_rng_state = None
            if torch.cuda.is_available():
                try:
                    cuda_rng_state = torch.cuda.get_rng_state_all()
                except Exception:
                    cuda_rng_state = None

            numpy_rng_state = np.random.get_state()
            python_rng_state = random.getstate()

            torch.save({
                # 模型权重
                "eval_qnet": pdqn.eval_qnet.state_dict(),
                "target_qnet": pdqn.target_qnet.state_dict(),
                "xnet": pdqn.xnet.state_dict(),

                # 优化器状态（无损续训必需）
                "optimizer_q": pdqn.optimizer_q.state_dict(),
                "optimizer_x": pdqn.optimizer_x.state_dict(),

                # 训练进度
                "episode": episode,
                "memory_counter": pdqn.memory_counter,
                "learn_step_counter": pdqn.learn_step_counter,

                # 随机性与可复现
                "seed": seed,
                "torch_rng_state": torch_rng_state,
                "cuda_rng_state": cuda_rng_state,
                "numpy_rng_state": numpy_rng_state,
                "python_rng_state": python_rng_state,
            }, os.path.join(save_dir, f"pdqn_ep{episode}.pt"))
end_time = time.time()
print("耗时间为：{}".format(end_time-start_time))