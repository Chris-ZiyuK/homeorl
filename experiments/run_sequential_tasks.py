"""
Sequential gridworld experiment for homeostatic RL.

This runner supports five agent families:
  A_task_only        : task reward, energy masked
  B_energy_aware     : task reward, energy observed
  C_task_homeostatic : task + homeostatic reward, energy observed
  D_pure_homeostatic : homeostatic reward, energy observed
  E_task_oracle      : task reward, energy observed, but reinitialized per task
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

from src.envs.sequential_homeostasis_env import SequentialHomeostasisEnv, TASK_SPECS


AGENT_SPECS = {
    "A_task_only": {
        "reward_mode": "task",
        "observation_mode": "masked",
        "sequential": True,
        "description": "Task reward only; energy hidden.",
    },
    "B_energy_aware": {
        "reward_mode": "task",
        "observation_mode": "full",
        "sequential": True,
        "description": "Task reward with energy observation.",
    },
    "C_task_homeostatic": {
        "reward_mode": "mixed",
        "observation_mode": "full",
        "sequential": True,
        "description": "Task reward plus homeostatic auxiliary reward.",
    },
    "D_pure_homeostatic": {
        "reward_mode": "homeostatic",
        "observation_mode": "full",
        "sequential": True,
        "description": "Pure homeostatic reward with sparse success/failure.",
    },
    "E_task_oracle": {
        "reward_mode": "task",
        "observation_mode": "full",
        "sequential": False,
        "description": "Task reward with energy observation; network reset per task.",
    },
}


DEFAULT_CONFIG = {
    "tasks": ["reach", "recharge", "hazard_reach", "detour", "tight_detour"],
    "agents": list(AGENT_SPECS.keys()),
    "episodes_per_task": 600,
    "eval_interval": 50,
    "eval_episodes": 40,
    "num_seeds": 10,
    "success_threshold": 0.7,
    "boundary_mode": "reset",
    "carryover_min_energy": 1.0,
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
    "environment": {
        "internal_coef": 1.0,
        "task_reward_coef": 1.0,
    },
    "logging": {
        "save_dir": "experiments/sequential_results",
        "plot_name": "sequential_agent_suite.png",
        "json_name": "sequential_agent_suite.json",
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


def make_env(task_name, agent_spec, env_cfg, start_energy=None):
    return SequentialHomeostasisEnv(
        task_name=task_name,
        reward_mode=agent_spec["reward_mode"],
        observation_mode=agent_spec["observation_mode"],
        internal_coef=env_cfg["internal_coef"],
        task_reward_coef=env_cfg["task_reward_coef"],
        initial_energy_override=start_energy,
    )


def init_training_stack(obs_dim, n_actions, agent_cfg):
    q_net = QNet(obs_dim, n_actions, hidden_dim=agent_cfg["hidden_dim"])
    target_net = QNet(obs_dim, n_actions, hidden_dim=agent_cfg["hidden_dim"])
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=agent_cfg["lr"])
    buffer = ReplayBuffer(capacity=agent_cfg["buffer_size"])
    return q_net, target_net, optimizer, buffer


def select_action(q_net, obs, eps, n_actions):
    if random.random() < eps:
        return random.randrange(n_actions)
    with torch.no_grad():
        return q_net(torch.FloatTensor(obs).unsqueeze(0)).argmax(1).item()


def evaluate_policy(q_net, task_name, agent_spec, cfg, n_eval=40, seed_base=100000, start_energy=None):
    successes = 0
    deaths = 0
    energy_left = 0.0
    hazard_hits = 0.0
    food_collected = 0.0

    for i in range(n_eval):
        env = make_env(task_name, agent_spec, cfg["environment"], start_energy=start_energy)
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


def resolve_phase_start_energy(task_name, cfg, previous_energy):
    if cfg["boundary_mode"] != "carryover" or previous_energy is None:
        return None
    task_cap = TASK_SPECS[task_name]["energy_cap"]
    return float(np.clip(previous_energy, cfg["carryover_min_energy"], task_cap))


def optimize_step(q_net, target_net, optimizer, buffer, agent_cfg):
    s, a, r, s2, d = buffer.sample(agent_cfg["batch_size"])
    q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        targets = r + agent_cfg["gamma"] * target_net(s2).max(1)[0] * (1 - d)
    loss = nn.MSELoss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_agent(agent_name, cfg, seed):
    agent_spec = AGENT_SPECS[agent_name]
    agent_cfg = cfg["agent"]
    tasks = cfg["tasks"]
    episodes_per_task = cfg["episodes_per_task"]
    eval_interval = cfg["eval_interval"]
    eval_episodes = cfg["eval_episodes"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    bootstrap_env = make_env(tasks[0], agent_spec, cfg["environment"])
    q_net, target_net, optimizer, buffer = init_training_stack(
        bootstrap_env.obs_dim, bootstrap_env.action_space.n, agent_cfg
    )

    logs = {
        "agent": agent_name,
        "seed": seed,
        "tasks": tasks,
        "boundary_mode": cfg["boundary_mode"],
        "agent_description": agent_spec["description"],
        "current_task_eval": [],
        "phase_end_eval": [],
    }

    global_episode = 0
    carryover_energy = None

    for task_idx, task_name in enumerate(tasks):
        if not agent_spec["sequential"]:
            q_net, target_net, optimizer, buffer = init_training_stack(
                bootstrap_env.obs_dim, bootstrap_env.action_space.n, agent_cfg
            )

        phase_start_energy = None if not agent_spec["sequential"] else resolve_phase_start_energy(
            task_name, cfg, carryover_energy
        )
        env = make_env(task_name, agent_spec, cfg["environment"], start_energy=phase_start_energy)
        buffer.clear()
        threshold_hit = None
        last_episode_energy = None

        for _ in range(episodes_per_task):
            obs, _ = env.reset(seed=seed * 100000 + global_episode)
            done = False
            while not done:
                eps = agent_cfg["eps_end"] + (
                    agent_cfg["eps_start"] - agent_cfg["eps_end"]
                ) * np.exp(-global_episode / agent_cfg["eps_decay"])
                action = select_action(q_net, obs, eps, env.action_space.n)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buffer.push(obs, action, reward, next_obs, float(done))
                obs = next_obs

                if len(buffer) >= agent_cfg["batch_size"]:
                    optimize_step(q_net, target_net, optimizer, buffer, agent_cfg)

            last_episode_energy = env.stats["energy_left"]

            if global_episode % agent_cfg["target_update"] == 0:
                target_net.load_state_dict(q_net.state_dict())

            global_episode += 1
            if global_episode % eval_interval == 0:
                metrics = evaluate_policy(
                    q_net,
                    task_name=task_name,
                    agent_spec=agent_spec,
                    cfg=cfg,
                    n_eval=eval_episodes,
                    seed_base=seed * 1000 + global_episode,
                    start_energy=phase_start_energy,
                )
                logs["current_task_eval"].append(
                    {
                        "global_episode": global_episode,
                        "task_index": task_idx,
                        "task_name": task_name,
                        "phase_start_energy": phase_start_energy,
                        **metrics,
                    }
                )
                if threshold_hit is None and metrics["success"] >= cfg["success_threshold"]:
                    threshold_hit = global_episode - task_idx * episodes_per_task

        task_matrix = {}
        for eval_task in tasks:
            eval_start_energy = phase_start_energy if eval_task == task_name else None
            task_matrix[eval_task] = evaluate_policy(
                q_net,
                task_name=eval_task,
                agent_spec=agent_spec,
                cfg=cfg,
                n_eval=eval_episodes,
                seed_base=seed * 2000 + global_episode + task_idx * 17,
                start_energy=eval_start_energy,
            )

        # --- Boundary solvability: probe next task from current terminal energy ---
        boundary_solvability = None
        boundary_solvability_energy = None
        if task_idx < len(tasks) - 1 and agent_spec["sequential"]:
            next_task = tasks[task_idx + 1]
            terminal_energy = task_matrix[task_name]["avg_energy_left"]
            if terminal_energy is not None and terminal_energy > 0:
                # Always compute probe energy as if carryover, to measure boundary safety
                next_cap = TASK_SPECS[next_task]["energy_cap"]
                probe_start = float(np.clip(
                    terminal_energy, cfg["carryover_min_energy"], next_cap
                ))
                probe_metrics = evaluate_policy(
                    q_net,
                    task_name=next_task,
                    agent_spec=agent_spec,
                    cfg=cfg,
                    n_eval=eval_episodes,
                    seed_base=seed * 3000 + global_episode + task_idx * 31,
                    start_energy=probe_start,
                )
                boundary_solvability = probe_metrics["success"]
                boundary_solvability_energy = probe_start
            else:
                boundary_solvability = 0.0
                boundary_solvability_energy = 0.0

        logs["phase_end_eval"].append(
            {
                "phase_task": task_name,
                "phase_index": task_idx,
                "adaptation_episode": threshold_hit,
                "phase_start_energy": phase_start_energy,
                "phase_end_energy": last_episode_energy,
                "policy_boundary_energy": task_matrix[task_name]["avg_energy_left"],
                "boundary_solvability": boundary_solvability,
                "boundary_solvability_energy": boundary_solvability_energy,
                "task_metrics": task_matrix,
            }
        )

        if agent_spec["sequential"]:
            carryover_energy = task_matrix[task_name]["avg_energy_left"]

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
                "phase_start_energy_mean": float(
                    np.nanmean([phase[phase_idx]["phase_start_energy"] for phase in phases])
                )
                if any(phase[phase_idx]["phase_start_energy"] is not None for phase in phases)
                else None,
                "phase_end_energy_mean": float(
                    np.nanmean([phase[phase_idx]["phase_end_energy"] for phase in phases])
                )
                if any(phase[phase_idx]["phase_end_energy"] is not None for phase in phases)
                else None,
                "policy_boundary_energy_mean": float(
                    np.nanmean([phase[phase_idx]["policy_boundary_energy"] for phase in phases])
                )
                if any(phase[phase_idx]["policy_boundary_energy"] is not None for phase in phases)
                else None,
                "adaptation_episode_mean": float(
                    np.nanmean(
                        [
                            phase[phase_idx]["adaptation_episode"]
                            if phase[phase_idx]["adaptation_episode"] is not None
                            else np.nan
                            for phase in phases
                        ]
                    )
                ),
                "boundary_solvability_mean": float(
                    np.nanmean(
                        [
                            phase[phase_idx].get("boundary_solvability", np.nan)
                            if phase[phase_idx].get("boundary_solvability") is not None
                            else np.nan
                            for phase in phases
                        ]
                    )
                ),
                "boundary_solvability_std": float(
                    np.nanstd(
                        [
                            phase[phase_idx].get("boundary_solvability", np.nan)
                            if phase[phase_idx].get("boundary_solvability") is not None
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
    agent_names = list(aggregated["current"].keys())
    cmap = plt.get_cmap("tab10")
    colors = {agent: cmap(idx % 10) for idx, agent in enumerate(agent_names)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for agent, series in aggregated["current"].items():
        x = [row["global_episode"] for row in series]
        y = [row["success_mean"] for row in series]
        std = [row["success_std"] for row in series]
        axes[0].plot(x, y, label=agent, color=colors[agent], linewidth=2.2)
        axes[0].fill_between(
            x,
            np.maximum(0.0, np.array(y) - np.array(std)),
            np.minimum(1.0, np.array(y) + np.array(std)),
            color=colors[agent],
            alpha=0.12,
        )

    for boundary in range(1, len(tasks)):
        axes[0].axvline(boundary * cfg["episodes_per_task"], color="#9ca3af", linestyle="--", linewidth=1.0)
    axes[0].set_title(f"Current Task Success ({cfg['boundary_mode']})")
    axes[0].set_xlabel("Global Episode")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    x_positions = np.arange(len(tasks))
    width = 0.82 / max(len(agent_names), 1)
    offsets = np.linspace(-0.41 + width / 2, 0.41 - width / 2, len(agent_names))
    for offset, agent in zip(offsets, agent_names):
        means = []
        stds = []
        for task_idx, task in enumerate(tasks):
            phase = aggregated["phase_end"][agent][task_idx]
            means.append(phase["task_metrics"][task]["success_mean"])
            stds.append(phase["task_metrics"][task]["success_std"])
        axes[1].bar(
            x_positions + offset,
            means,
            width=width,
            yerr=stds,
            label=agent,
            color=colors[agent],
            alpha=0.85,
        )

    axes[1].set_title("End-of-Phase Performance per Task")
    axes[1].set_xticks(x_positions, tasks)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Success Rate")
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
    parser.add_argument("--seed_index", type=int, default=None,
                        help="Run only this seed index (0-based). For SLURM array dispatch.")
    parser.add_argument("--boundary_mode", type=str, default=None,
                        choices=["reset", "carryover"],
                        help="Override boundary_mode from config.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory.")
    args = parser.parse_args()
    cfg = load_config(args.config if os.path.exists(args.config) else None)

    if args.boundary_mode:
        cfg["boundary_mode"] = args.boundary_mode
    if args.output_dir:
        cfg["logging"]["save_dir"] = args.output_dir

    # Determine which seeds to run
    if args.seed_index is not None:
        seed_indices = [args.seed_index]
    else:
        seed_indices = list(range(cfg["num_seeds"]))

    raw_runs = []
    for seed_idx in seed_indices:
        seed = 100 + seed_idx * 17
        for agent_name in cfg["agents"]:
            print(f"Running {agent_name:20s} seed={seed} boundary={cfg['boundary_mode']}")
            raw_runs.append(train_agent(agent_name, cfg, seed))

    aggregated = aggregate_runs(raw_runs, cfg["tasks"])

    # For single-seed runs, use seed-specific filenames
    if args.seed_index is not None:
        base_name = Path(cfg["logging"]["json_name"]).stem
        cfg["logging"]["json_name"] = f"{base_name}_seed{args.seed_index}.json"
        cfg["logging"]["plot_name"] = f"{base_name}_seed{args.seed_index}.png"

    plot_path = plot_results(aggregated, cfg)
    json_path = save_results(raw_runs, aggregated, cfg)

    print("\nFinal task-wise success means:")
    for agent in cfg["agents"]:
        final_phase = aggregated["phase_end"][agent][-1]
        print(f"  {agent}")
        for task_name in cfg["tasks"]:
            success = final_phase["task_metrics"][task_name]["success_mean"]
            print(f"    {task_name:<12s} success={success:.3f}")

    print(f"\nSaved plot: {plot_path}")
    print(f"Saved data: {json_path}")


if __name__ == "__main__":
    main()
