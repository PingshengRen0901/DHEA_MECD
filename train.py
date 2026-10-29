"""Unified training/evaluation for DHEA-MECD, seven baselines and ablations."""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from algorithms.baselines import BaselineAgent, BaselineConfig
from config import AgentConfig, ObservationLayout
from hyenv import MovingEnv
from hyenv.environment import RewardConfig
from hyenv.observations import FEATURES
from pdqn import PDQN

ALGORITHMS = ("dhea_mecd", "pdqn", "ppo", "sac", "ddpg", "trpo", "ac", "dqn")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON experiment configuration")
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="dhea_mecd")
    parser.add_argument("--dimension", choices=["9", "18", "27", "36", "full"], default="36")
    parser.add_argument("--ablation", choices=["none", "no_encoder_attention", "no_mecd"], default="none")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--obstacles", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=200_000)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target-update", choices=["soft", "hard"], default="soft")
    parser.add_argument("--uniform-replay", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/dhea_mecd"))
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--evaluation-seed", type=int, default=1_000_000)
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.checkpoint_every < 1 or min(args.warmup, args.latency_warmup) < 0:
        parser.error("Positive episodes/checkpoint period and nonnegative warmup required")
    if args.resume and args.evaluate:
        parser.error("Choose --resume or --evaluate")
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_agent(path, device="cpu"):
    kind = torch.load(path, map_location="cpu", weights_only=False).get("kind", "dhea_mecd")
    return (BaselineAgent if kind == "baseline" else PDQN).load(path, device)


def build_run(args):
    supplied = json.loads(args.config.read_text()) if args.config else {}
    algorithm = supplied.get("algorithm", args.algorithm)
    if algorithm not in ALGORITHMS:
        raise ValueError("Unsupported algorithm")
    training = {"seed": args.seed, "max_steps": args.max_steps, "warmup": args.warmup,
                **supplied.get("training", {})}
    dimension = None if args.dimension == "full" else int(args.dimension)
    env_config = {"n_targets": args.targets, "n_obstacles": args.obstacles, "max_step": training["max_steps"],
                  "observation_dim": dimension, **supplied.get("environment", {})}
    config = AgentConfig(**{"device": args.device, "batch_size": args.batch_size,
                            "capacity": args.capacity, "sequence_length": args.sequence_length,
                            "tau": args.tau, "prioritized": not args.uniform_replay,
                            "top_k": args.top_k, "target_update": args.target_update,
                            **supplied.get("agent", {})})
    config.device = args.device
    ablation = supplied.get("ablation", args.ablation)
    if ablation not in ("none", "no_encoder_attention", "no_mecd"):
        raise ValueError("Unknown ablation")
    if ablation != "none" and algorithm != "dhea_mecd":
        raise ValueError("Component ablations are defined for DHEA-MECD")
    config.use_encoder_attention = ablation != "no_encoder_attention"
    config.use_moe = algorithm == "dhea_mecd" and ablation != "no_mecd"
    # Baseline PDQN differs in its single Q head and uniform replay.
    if algorithm != "dhea_mecd":
        config.prioritized = False
    baseline = BaselineConfig(**supplied.get("baseline", {}))
    if training["max_steps"] != env_config["max_step"]:
        raise ValueError("training.max_steps must match environment.max_step")
    seed_all(training["seed"])
    env = make_env(env_config, args.render)
    if algorithm in ("dhea_mecd", "pdqn"):
        agent = PDQN(env.layout, config, seed=training["seed"])
    else:
        agent = BaselineAgent(algorithm, env.layout, config, baseline, seed=training["seed"])
    metadata = {"algorithm": algorithm, "ablation": ablation, "environment": env_config,
                "training": training, "baseline": asdict(baseline)}
    return agent, env, metadata


def make_env(config, render=False):
    config = dict(config)
    if isinstance(config.get("reward_config"), dict):
        config["reward_config"] = RewardConfig(**config["reward_config"])
    return MovingEnv(**config, render_mode="human" if render else None)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elif torch.device(device).type == "mps":
        torch.mps.synchronize()


def main():
    args = parse_args()
    seed_all(args.seed)
    start_episode = 0
    if args.resume or args.evaluate:
        agent, completed = load_agent(args.resume or args.evaluate, args.device)
        metadata = agent.metadata
        if not metadata:
            raise ValueError("Checkpoint lacks environment metadata; use a newly generated checkpoint")
        env = make_env(metadata["environment"], args.render)
        if args.resume:
            start_episode = completed + 1
    else:
        agent, env, metadata = build_run(args)
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = "evaluation" if args.evaluate else "training"
    run_config = {**metadata, "agent": asdict(agent.config), "layout": asdict(agent.layout),
                  "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                              "machine": platform.machine(), "torch": torch.__version__, "numpy": np.__version__,
                              "threads": torch.get_num_threads(), "device": args.device,
                              "cuda_device": torch.cuda.get_device_name() if torch.device(args.device).type == "cuda" else None},
                  "feature_names": FEATURES.get(agent.layout.observation_dim),
                  "length_unit": "m" if env.length_scale_m else "simulation_unit",
                  "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    config_path = args.output / f"{prefix}_config.json"
    metrics_path = args.output / f"{prefix}.csv"
    if metrics_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to append a new run to {metrics_path}; choose a fresh output directory")
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    fields = ["episode", "reward", "steps", "success", "captured_targets", "collision", "boundary",
              "trajectory_length", "energy_cost", "epsilon", "loss_q", "loss_actor", "seconds",
              "inference_mean_ms", "inference_p50_ms", "inference_p95_ms", "terminated", "truncated"]
    existing = metrics_path.exists() and metrics_path.stat().st_size > 0
    summary_rows = []
    try:
        # Exclude cold-start inference from latency measurements.
        if args.evaluate:
            state, _ = env.reset(seed=args.evaluation_seed)
            for _ in range(args.latency_warmup):
                agent.choose_action(state, explore=False)
            agent.reset_episode()
        with metrics_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if not existing:
                writer.writeheader()
            for episode in range(start_episode, start_episode + args.episodes):
                started = time.perf_counter()
                reset_seed = args.evaluation_seed + episode if args.evaluate else metadata["training"]["seed"] + episode
                state, _ = env.reset(seed=reset_seed)
                agent.reset_episode()
                total_reward, losses, latencies = 0.0, None, []
                auv_path, target_path, obstacle_path = [env.agent.position.copy()], [env.targets.copy()], [env.obstacles.copy()]
                for step in range(1, env.max_step + 1):
                    synchronize(args.device)
                    inference_start = time.perf_counter()
                    action, parameters = agent.choose_action(state, episode, explore=not args.evaluate)
                    synchronize(args.device)
                    latencies.append((time.perf_counter() - inference_start)*1000)
                    next_state, reward, terminated, truncated, info = env.step((int(action[0]), parameters[0]))
                    if not args.evaluate:
                        agent.store_transition(state, action[0], parameters[0], reward, next_state, terminated, truncated)
                        on_policy = isinstance(agent, BaselineAgent) and agent.algorithm in ("ppo", "ac", "trpo")
                        if on_policy or agent.memory_counter >= max(metadata["training"]["warmup"], agent.config.batch_size):
                            update = agent.learn()
                            if update is not None:
                                losses = update
                    state = next_state
                    total_reward += reward
                    if args.save_trajectories:
                        auv_path.append(env.agent.position.copy())
                        target_path.append(env.targets.copy())
                        obstacle_path.append(env.obstacles.copy())
                    if args.render:
                        env.render()
                    if terminated or truncated:
                        break
                if not args.evaluate and isinstance(agent, BaselineAgent) and agent.algorithm in ("ppo", "ac", "trpo"):
                    update = agent.learn(force=True)
                    if update is not None:
                        losses = update
                row = {"episode": episode, "reward": total_reward, "steps": step, **info,
                       "epsilon": 0 if args.evaluate else agent.epsilon(episode),
                       "loss_q": losses[0] if losses else "", "loss_actor": losses[1] if losses else "",
                       "seconds": time.perf_counter() - started, "inference_mean_ms": np.mean(latencies),
                       "inference_p50_ms": np.median(latencies), "inference_p95_ms": np.percentile(latencies, 95),
                       "terminated": terminated, "truncated": truncated}
                row.pop("reward_components", None)
                if env.length_scale_m:
                    row["trajectory_length"] *= env.length_scale_m
                writer.writerow(row)
                file.flush()
                summary_rows.append(row)
                if args.save_trajectories:
                    directory = args.output / "trajectories"
                    directory.mkdir(exist_ok=True)
                    np.savez_compressed(directory / f"episode_{episode}.npz", auv=auv_path, targets=target_path,
                                        obstacles=obstacle_path, captured=env.captured, dt=env.delta_t,
                                        agent_radius=env.agent_radius, target_radius=env.target_radius,
                                        obstacle_radius=env.obstacle_radius, field_size=env.field_size)
                print(f"{metadata['algorithm']} episode={episode} reward={total_reward:.4f} steps={step} "
                      f"captured={info['captured_targets']}/{env.n_targets} success={info['success']} "
                      f"updates={agent.learn_step_counter}", flush=True)
                if not args.evaluate and ((episode + 1) % args.checkpoint_every == 0 or
                                          episode == start_episode + args.episodes - 1):
                    agent.save(args.output / "latest.pt", episode, metadata)
        if args.evaluate:
            summary = {"episodes": len(summary_rows), "success_rate": np.mean([r["success"] for r in summary_rows]),
                       "collision_rate": np.mean([r["collision"] for r in summary_rows]),
                       "mean_energy": np.mean([r["energy_cost"] for r in summary_rows]),
                       "sd_energy": np.std([r["energy_cost"] for r in summary_rows], ddof=1) if len(summary_rows)>1 else None,
                       "latency_device": args.device, "latency_includes": "host transfer, history, encoder, policy; excludes env.step"}
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
