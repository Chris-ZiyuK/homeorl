"""
Poison Food Experiment — DQN + 4 Query Strategies
===================================================
Run: conda activate minihack && python run_experiment.py

Groups:
  B1 (No-QUERY):      Never queries
  B2 (Always-QUERY):  Queries on first step
  B3 (Random-QUERY):  50% chance to query
  G1 (Gated-QUERY):   p(QUERY) = sigmoid(alpha*(E*-E-eps))
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poison_food_env import PoisonFoodEnv

# ═══════════════════════════════════════════════════════════════
# DQN Network
# ═══════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    def __init__(self, obs_dim=11, n_actions=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

# ═══════════════════════════════════════════════════════════════
# Replay Buffer
# ═══════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

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

    def __len__(self):
        return len(self.buffer)

# ═══════════════════════════════════════════════════════════════
# Query Policies
# ═══════════════════════════════════════════════════════════════

QUERY_ACTION = 4

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def gated_query_prob(energy, E_init=80, E_star=75, alpha=0.3, epsilon=0):
    """Gate triggers early: at E=80 (full), p≈0.18; at E=72 (one step in), p≈0.71"""
    return sigmoid(alpha * (E_star - energy - epsilon))

def apply_query_policy(policy_name, env, has_queried):
    """Returns (action_override, updated_has_queried)."""
    if has_queried:
        return None, True

    if policy_name == "B1_NoQuery":
        return None, False

    elif policy_name == "B2_AlwaysQuery":
        return QUERY_ACTION, True

    elif policy_name == "B3_RandomQuery":
        if random.random() < 0.5:
            return QUERY_ACTION, True
        return None, False

    elif policy_name == "G1_GatedQuery":
        p = gated_query_prob(env.energy)
        if random.random() < p:
            return QUERY_ACTION, True
        return None, False

    return None, False

# ═══════════════════════════════════════════════════════════════
# Train one policy
# ═══════════════════════════════════════════════════════════════

def train_policy(policy_name, n_episodes=3000, lr=5e-4, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=800,
                 batch_size=64, target_update=30, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = PoisonFoodEnv(reward_type='hrrl')
    # DQN only controls 4 movement actions (QUERY is policy-controlled)
    q_net = QNetwork(obs_dim=11, n_actions=4, hidden=128)
    target_net = QNetwork(obs_dim=11, n_actions=4, hidden=128)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    rewards_history = []
    survival_history = []
    query_history = []
    exit_history = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0
        has_queried = False

        for t in range(env.max_steps):
            # Query policy decides whether to QUERY
            override, has_queried = apply_query_policy(
                policy_name, env, has_queried)

            if override is not None:
                action = override
            else:
                # DQN epsilon-greedy (only movement actions 0-3)
                eps = eps_end + (eps_start - eps_end) * \
                      np.exp(-ep / eps_decay)
                if random.random() < eps:
                    action = random.randint(0, 3)
                else:
                    with torch.no_grad():
                        q_vals = q_net(torch.FloatTensor(obs).unsqueeze(0))
                        action = q_vals.argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Only store movement experiences for DQN learning
            if action < 4:
                buffer.push(obs, action, reward, next_obs, float(done))

            obs = next_obs

            # Train DQN
            if len(buffer) >= batch_size:
                states, actions, rewards_b, next_states, dones = \
                    buffer.sample(batch_size)
                q_values = q_net(states).gather(1, actions.unsqueeze(1))
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    targets = rewards_b + gamma * next_q * (1 - dones)
                loss = nn.MSELoss()(q_values.squeeze(), targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        if ep % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        rewards_history.append(total_reward)
        survival_history.append(0.0 if env.stats["energy_depleted"] else 1.0)
        query_history.append(env.stats["queries"])
        exit_history.append(1.0 if env.stats["reached_exit"] else 0.0)

        if (ep + 1) % 500 == 0:
            w = 200
            avg_r = np.mean(rewards_history[-w:])
            avg_s = np.mean(survival_history[-w:])
            avg_e = np.mean(exit_history[-w:])
            avg_q = np.mean(query_history[-w:])
            print(f"  [{policy_name:16s}] Ep {ep+1:4d}  "
                  f"R={avg_r:+.3f}  Surv={avg_s:.0%}  "
                  f"Exit={avg_e:.0%}  Q={avg_q:.1f}")

    return {
        "name": policy_name,
        "rewards": rewards_history,
        "survival": survival_history,
        "exit_rate": exit_history,
        "queries": query_history,
    }

# ═══════════════════════════════════════════════════════════════
# Plotting (English labels to avoid font issues)
# ═══════════════════════════════════════════════════════════════

def smooth(data, window=100):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode="valid")

def plot_results(results, save_dir="."):
    colors = {
        "B1_NoQuery": "#ef4444",
        "B2_AlwaysQuery": "#f59e0b",
        "B3_RandomQuery": "#8b5cf6",
        "G1_GatedQuery": "#10b981",
    }
    labels = {
        "B1_NoQuery": "B1: No Query",
        "B2_AlwaysQuery": "B2: Always Query",
        "B3_RandomQuery": "B3: Random Query",
        "G1_GatedQuery": "G1: Gated Query (ours)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Poison Food Experiment — Query Strategy Comparison",
                 fontsize=14, fontweight="bold")

    metrics = [
        ("rewards",   "Cumulative Reward",  axes[0, 0]),
        ("survival",  "Survival Rate",      axes[0, 1]),
        ("exit_rate", "Exit Reached Rate",  axes[1, 0]),
        ("queries",   "Queries per Episode", axes[1, 1]),
    ]

    for key, title, ax in metrics:
        for r in results:
            name = r["name"]
            data = smooth(r[key])
            ax.plot(data, color=colors[name], label=labels[name],
                    linewidth=2, alpha=0.85)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Episode")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "experiment_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: {path}")
    return path

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Poison Food Experiment — Energy-Gated QUERY Validation (HRRL Driven)")
    print("=" * 60)
    print(f"Env: 7x7 grid, E_init=80, c_step=8, c_query=5")
    print(f"Agent MUST eat safe food to survive. Reward strictly internal drive reduction.")
    print(f"Poison food = instant death, QUERY reveals which is safe")
    print()

    policies = ["B1_NoQuery", "B2_AlwaysQuery",
                "B3_RandomQuery", "G1_GatedQuery"]
    all_results = []

    for policy in policies:
        print(f"\n>>> Training: {policy}")
        result = train_policy(policy)
        all_results.append(result)

    # Final stats
    print("\n" + "=" * 60)
    print("Final Results (last 300 episodes avg)")
    print("=" * 60)
    print(f"{'Policy':<18s} {'Reward':>8s} {'Survive':>8s} {'Exit':>8s} {'Queries':>8s}")
    print("-" * 52)
    for r in all_results:
        w = 300
        avg_r = np.mean(r["rewards"][-w:])
        avg_s = np.mean(r["survival"][-w:])
        avg_e = np.mean(r["exit_rate"][-w:])
        avg_q = np.mean(r["queries"][-w:])
        print(f"{r['name']:<18s} {avg_r:>+8.3f} {avg_s:>7.1%} "
              f"{avg_e:>7.1%} {avg_q:>8.2f}")

    # Plot
    save_dir = os.path.dirname(os.path.abspath(__file__))
    plot_results(all_results, save_dir=save_dir)
    print("\nDone!")

if __name__ == "__main__":
    main()
