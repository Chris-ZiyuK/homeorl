"""
Sequential gridworld experiment for homeostatic RL.

Compares two agents trained on the same task sequence:
  - task:        dense task-specific shaping that changes with each task
  - homeostatic: shared drive-reduction objective across all tasks
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.sequential_homeostasis_env import SequentialHomeostasisEnv


DEFAULT_CONFIG = {
    "tasks": ["reach", "recharge", "hazard_reach", "detour", "tight_detour"],
    "episodes_per_task": 600,
    "eval_interval": 50,
    "eval_episodes": 40,
    "num_seeds": 10,
    "success_threshold": 0.7,
    "agent": {
        "hidden_dim": 128,
        "lr": 5e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "buffer_size": 20000,
        "target_update": 25,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay": 300,
    },
    "logging": {
        "save_dir": "experiments/sequential_results",
        "plot_name": "sequential_homeostasis.png",
        "json_name": "sequential_homeostasis.json",
    },
}


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


def merge_dict(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path):
    if not config_path:
        return DEFAULT_CONFIG
    if yaml is None:
        raise RuntimeError("pyyaml is required to load config files")
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return merge_dict(DEFAULT_CONFIG, loaded)


def evaluate_policy(q_net, task_name, reward_mode, n_eval=40, seed_base=100000):
    successes = 0
    deaths = 0
    energy_left = 0.0
    hazard_hits = 0.0
    food_collected = 0.0

    for i in range(n_eval):
        env = SequentialHomeostasisEnv(task_name=task_name, reward_mode=reward_mode)
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        while not done:
            with torch.no_grad():
                action = q_net(torch.FloatTensor(obs).unsqueeze(0)).argmax(1).item()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        successes += int(env.stats["success"])
        deaths += int(env.stats["energy_depleted"])
        energy_left += env.stats["energy_left"]
        hazard_hits += env.stats["hazard_hits"]
        food_collected += int(env.stats["food_collected"])

    return {
        "success": successes / n_eval,
        "death": deaths / n_eval,
        "avg_energy_left": energy_left / n_eval,
        "avg_hazard_hits": hazard_hits / n_eval,
        "food_rate": food_collected / n_eval,
    }


def train_agent(agent_name, cfg, seed):
    agent_cfg = cfg["agent"]
    tasks = cfg["tasks"]
    episodes_per_task = cfg["episodes_per_task"]
    eval_interval = cfg["eval_interval"]
    eval_episodes = cfg["eval_episodes"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    reward_mode = "task" if agent_name == "task" else "homeostatic"
    env = SequentialHomeostasisEnv(task_name=tasks[0], reward_mode=reward_mode)
    q_net = QNet(env.obs_dim, env.action_space.n, hidden_dim=agent_cfg["hidden_dim"])
    target_net = QNet(env.obs_dim, env.action_space.n, hidden_dim=agent_cfg["hidden_dim"])
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=agent_cfg["lr"])
    buffer = ReplayBuffer(capacity=agent_cfg["buffer_size"])

    logs = {
        "agent": agent_name,
        "seed": seed,
        "tasks": tasks,
        "current_task_eval": [],
        "phase_end_eval": [],
    }

    global_episode = 0
    for task_idx, task_name in enumerate(tasks):
        env = SequentialHomeostasisEnv(task_name=task_name, reward_mode=reward_mode)
        buffer.clear()
        threshold_hit = None

        for local_episode in range(episodes_per_task):
            obs, _ = env.reset(seed=seed * 100000 + global_episode)
            done = False
            while not done:
                eps = agent_cfg["eps_end"] + (
                    agent_cfg["eps_start"] - agent_cfg["eps_end"]
                ) * np.exp(-global_episode / agent_cfg["eps_decay"])
                if random.random() < eps:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = q_net(torch.FloatTensor(obs).unsqueeze(0)).argmax(1).item()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buffer.push(obs, action, reward, next_obs, float(done))
                obs = next_obs

                if len(buffer) >= agent_cfg["batch_size"]:
                    s, a, r, s2, d = buffer.sample(agent_cfg["batch_size"])
                    q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        targets = r + agent_cfg["gamma"] * target_net(s2).max(1)[0] * (1 - d)
                    loss = nn.MSELoss()(q_values, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if global_episode % agent_cfg["target_update"] == 0:
                target_net.load_state_dict(q_net.state_dict())

            global_episode += 1
            if global_episode % eval_interval == 0:
                metrics = evaluate_policy(
                    q_net,
                    task_name=task_name,
                    reward_mode=reward_mode,
                    n_eval=eval_episodes,
                    seed_base=seed * 1000 + global_episode,
                )
                logs["current_task_eval"].append(
                    {
                        "global_episode": global_episode,
                        "task_index": task_idx,
                        "task_name": task_name,
                        **metrics,
                    }
                )
                if threshold_hit is None and metrics["success"] >= cfg["success_threshold"]:
                    threshold_hit = global_episode - task_idx * episodes_per_task

        task_matrix = {}
        for eval_task in tasks:
            task_matrix[eval_task] = evaluate_policy(
                q_net,
                task_name=eval_task,
                reward_mode=reward_mode,
                n_eval=eval_episodes,
                seed_base=seed * 2000 + global_episode + task_idx * 17,
            )
        logs["phase_end_eval"].append(
            {
                "phase_task": task_name,
                "phase_index": task_idx,
                "adaptation_episode": threshold_hit,
                "task_metrics": task_matrix,
            }
        )

    return logs


def aggregate_runs(runs, tasks):
    current_by_agent = {}
    phase_by_agent = {}

    for run in runs:
        agent = run["agent"]
        current_by_agent.setdefault(agent, [])
        phase_by_agent.setdefault(agent, [])
        current_by_agent[agent].append(run["current_task_eval"])
        phase_by_agent[agent].append(run["phase_end_eval"])

    aggregated = {"current": {}, "phase_end": {}}

    for agent, logs in current_by_agent.items():
        n_points = len(logs[0])
        series = []
        for idx in range(n_points):
            point = {
                "global_episode": logs[0][idx]["global_episode"],
                "task_name": logs[0][idx]["task_name"],
                "task_index": logs[0][idx]["task_index"],
            }
            for metric in ["success", "death", "avg_energy_left", "avg_hazard_hits", "food_rate"]:
                vals = [log[idx][metric] for log in logs]
                point[f"{metric}_mean"] = float(np.mean(vals))
                point[f"{metric}_std"] = float(np.std(vals))
            series.append(point)
        aggregated["current"][agent] = series

    for agent, phases in phase_by_agent.items():
        phase_rows = []
        n_phases = len(phases[0])
        for phase_idx in range(n_phases):
            row = {
                "phase_task": phases[0][phase_idx]["phase_task"],
                "phase_index": phase_idx,
                "adaptation_episode_mean": float(
                    np.mean(
                        [
                            phase[phase_idx]["adaptation_episode"]
                            if phase[phase_idx]["adaptation_episode"] is not None
                            else np.nan
                            for phase in phases
                        ]
                    )
                ),
                "task_metrics": {},
            }
            for task_name in tasks:
                row["task_metrics"][task_name] = {}
                for metric in ["success", "death", "avg_energy_left", "avg_hazard_hits", "food_rate"]:
                    vals = [phase[phase_idx]["task_metrics"][task_name][metric] for phase in phases]
                    row["task_metrics"][task_name][f"{metric}_mean"] = float(np.mean(vals))
                    row["task_metrics"][task_name][f"{metric}_std"] = float(np.std(vals))
            phase_rows.append(row)
        aggregated["phase_end"][agent] = phase_rows

    return aggregated


def plot_results(aggregated, cfg):
    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / cfg["logging"]["plot_name"]
    tasks = cfg["tasks"]
    colors = {"task": "#ef4444", "homeostatic": "#2563eb"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for agent, series in aggregated["current"].items():
        x = [row["global_episode"] for row in series]
        y = [row["success_mean"] for row in series]
        std = [row["success_std"] for row in series]
        axes[0].plot(x, y, label=agent, color=colors[agent], linewidth=2.5)
        axes[0].fill_between(
            x,
            np.maximum(0.0, np.array(y) - np.array(std)),
            np.minimum(1.0, np.array(y) + np.array(std)),
            color=colors[agent],
            alpha=0.18,
        )

    for boundary in range(1, len(tasks)):
        axes[0].axvline(boundary * cfg["episodes_per_task"], color="#9ca3af", linestyle="--")
    axes[0].set_title("Current Task Success During Sequential Training")
    axes[0].set_xlabel("Global Episode")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    x_positions = np.arange(len(tasks))
    width = 0.18
    for offset, agent in zip([-0.18, 0.18], ["task", "homeostatic"]):
        final_phase = aggregated["phase_end"][agent][-1]
        means = [
            final_phase["task_metrics"][task]["success_mean"]
            for task in tasks
        ]
        stds = [
            final_phase["task_metrics"][task]["success_std"]
            for task in tasks
        ]
        axes[1].bar(
            x_positions + offset,
            means,
            width=width * 2,
            yerr=stds,
            label=agent,
            color=colors[agent],
            alpha=0.85,
        )

    axes[1].set_title("Final Retention Across All Tasks")
    axes[1].set_xticks(x_positions, tasks)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Success Rate")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def save_results(raw_runs, aggregated, cfg):
    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / cfg["logging"]["json_name"]
    payload = {"config": cfg, "aggregated": aggregated, "raw_runs": raw_runs}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sequential_gridworld.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config if os.path.exists(args.config) else None)

    raw_runs = []
    for seed_idx in range(cfg["num_seeds"]):
        seed = 100 + seed_idx * 17
        for agent_name in ["task", "homeostatic"]:
            print(f"Running {agent_name:12s} seed={seed}")
            raw_runs.append(train_agent(agent_name, cfg, seed))

    aggregated = aggregate_runs(raw_runs, cfg["tasks"])
    plot_path = plot_results(aggregated, cfg)
    json_path = save_results(raw_runs, aggregated, cfg)

    print("\nPhase-end success means:")
    for agent in ["task", "homeostatic"]:
        final_phase = aggregated["phase_end"][agent][-1]
        print(f"  {agent}")
        for task_name in cfg["tasks"]:
            success = final_phase["task_metrics"][task_name]["success_mean"]
            print(f"    {task_name:<10s} success={success:.3f}")

    print(f"\nSaved plot: {plot_path}")
    print(f"Saved data: {json_path}")


if __name__ == "__main__":
    main()
