"""
Homeostatic Grounding Experiment
=================================
Run: conda activate minihack && python grounding_experiment.py

Tests whether internal energy observation accelerates symbol grounding:
  Group A (Terminal-Only): agent has no energy info, learns only from death/exit
  Group B (Energy-Aware):  agent observes energy, gets immediate feedback on objects
  Group C (Energy+QUERY):  agent observes energy + can query for object hints
"""

import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# from multi_object_env import MultiObjectEnv
from minihack_grounding_env import MiniHackGroundingEnv as MultiObjectEnv


# ═══════════ DQN ═══════════

class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, cap=20000):
        self.buf = deque(maxlen=cap)
    def push(self, *args):
        self.buf.append(args)
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.LongTensor(a),
                torch.FloatTensor(r),
                torch.FloatTensor(np.array(s2)),
                torch.FloatTensor(d))
    def __len__(self):
        return len(self.buf)

# ═══════════ FGS Evaluation ═══════════

def evaluate_fgs(env_cls, obs_mode, q_net, n_eval=80, seed_base=99999):
    """
    Functional Grounding Score:
      FGS = P(approach beneficial) - P(approach harmful)
    
    We measure: did the agent eat beneficial objects and avoid harmful ones?
    """
    ate_beneficial = 0
    ate_harmful = 0
    total_beneficial_present = 0
    total_harmful_present = 0
    survived = 0
    reached_exit = 0

    for i in range(n_eval):
        env = env_cls(obs_mode=obs_mode)
        obs, _ = env.reset(seed=seed_base + i)
        done = False

        while not done:
            with torch.no_grad():
                q = q_net(torch.FloatTensor(obs).unsqueeze(0))
                if obs_mode != 'full':
                    action = q.argmax(1).item()
                else:
                    action = q.argmax(1).item()
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc

        # Count outcomes
        total_beneficial_present += 1  # always 1 beneficial per episode
        total_harmful_present += 1     # always 1 harmful per episode

        if env.stats["ate_beneficial"] > 0:
            ate_beneficial += 1
        if env.stats["ate_harmful"] > 0:
            ate_harmful += 1
        if not env.stats["energy_depleted"]:
            survived += 1
        if env.stats["reached_exit"]:
            reached_exit += 1

    p_approach_good = ate_beneficial / max(total_beneficial_present, 1)
    p_approach_bad = ate_harmful / max(total_harmful_present, 1)
    fgs = p_approach_good - p_approach_bad

    return {
        "fgs": fgs,
        "p_eat_beneficial": p_approach_good,
        "p_eat_harmful": p_approach_bad,
        "survival_rate": survived / n_eval,
        "exit_rate": reached_exit / n_eval,
    }

# ═══════════ Training ═══════════

def train_group(group_name, obs_mode, n_episodes=3000,
                lr=5e-4, gamma=0.99, batch_size=64,
                eps_start=1.0, eps_end=0.05, eps_decay=600,
                target_update=25, eval_interval=100, seed=42):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = MultiObjectEnv(obs_mode=obs_mode)
    obs_dim = env.obs_dim
    n_act = env.action_space.n

    q_net = QNet(obs_dim, n_act)
    target_net = QNet(obs_dim, n_act)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buf = ReplayBuffer()

    # Logs
    fgs_history = []
    survival_history = []
    exit_history = []
    eat_good_history = []
    eat_bad_history = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0

        for t in range(env.max_steps):
            eps = eps_end + (eps_start - eps_end) * np.exp(-ep / eps_decay)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q_net(torch.FloatTensor(obs).unsqueeze(0)).argmax(1).item()

            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            total_reward += reward
            buf.push(obs, action, reward, next_obs, float(done))
            obs = next_obs

            if len(buf) >= batch_size:
                s, a, r, s2, d = buf.sample(batch_size)
                qv = q_net(s).gather(1, a.unsqueeze(1))
                with torch.no_grad():
                    nq = target_net(s2).max(1)[0]
                    tgt = r + gamma * nq * (1 - d)
                loss = nn.MSELoss()(qv.squeeze(), tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        if ep % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Periodic evaluation
        if (ep + 1) % eval_interval == 0:
            metrics = evaluate_fgs(MultiObjectEnv, obs_mode, q_net)
            fgs_history.append(metrics["fgs"])
            survival_history.append(metrics["survival_rate"])
            exit_history.append(metrics["exit_rate"])
            eat_good_history.append(metrics["p_eat_beneficial"])
            eat_bad_history.append(metrics["p_eat_harmful"])

            if (ep + 1) % 500 == 0:
                print(f"  [{group_name:22s}] Ep {ep+1:4d}  "
                      f"FGS={metrics['fgs']:+.3f}  "
                      f"Surv={metrics['survival_rate']:.0%}  "
                      f"Exit={metrics['exit_rate']:.0%}  "
                      f"Eat+={metrics['p_eat_beneficial']:.0%}  "
                      f"Eat-={metrics['p_eat_harmful']:.0%}")

    return {
        "name": group_name,
        "fgs": fgs_history,
        "survival": survival_history,
        "exit_rate": exit_history,
        "eat_beneficial": eat_good_history,
        "eat_harmful": eat_bad_history,
        "eval_episodes": list(range(eval_interval, n_episodes + 1, eval_interval)),
    }

# ═══════════ Plotting ═══════════

def plot_results(results, save_dir="."):
    colors = {
        "A: Terminal-Only": "#ef4444",
        "B: Energy-Aware":  "#f59e0b",
        "C: Energy+QUERY":  "#10b981",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Homeostatic Grounding Experiment", fontsize=15, fontweight="bold")

    plots = [
        ("fgs",            "Functional Grounding Score (FGS)", axes[0, 0]),
        ("survival",       "Survival Rate",                    axes[0, 1]),
        ("eat_beneficial", "P(eat beneficial object)",         axes[1, 0]),
        ("eat_harmful",    "P(eat harmful object)",            axes[1, 1]),
    ]

    for key, title, ax in plots:
        for r in results:
            eps = r["eval_episodes"]
            data = r[key]
            ax.plot(eps, data, color=colors[r["name"]], label=r["name"],
                    linewidth=2.5, alpha=0.85)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Training Episode")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if key == "fgs":
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylabel("FGS = P(eat good) - P(eat bad)")

    plt.tight_layout()
    path = os.path.join(save_dir, "grounding_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: {path}")
    return path

# ═══════════ Main ═══════════

def main():
    print("=" * 60)
    print("Homeostatic Grounding Experiment")
    print("Does internal energy help agents learn object meanings?")
    print("=" * 60)
    print("Objects: Type 0 (harmful, E-=40) | Type 1 (beneficial, E+=30) | Type 2 (neutral)")
    print("Groups:  A=terminal-only | B=energy-aware | C=energy+QUERY")
    print()

    groups = [
        ("A: Terminal-Only", "terminal"),
        ("B: Energy-Aware",  "energy"),
        ("C: Energy+QUERY",  "full"),
    ]

    all_results = []
    for name, mode in groups:
        print(f"\n>>> Training: {name}")
        result = train_group(name, mode)
        all_results.append(result)

    # Final summary
    print("\n" + "=" * 60)
    print("Final Results (last 5 evaluations)")
    print("=" * 60)
    print(f"{'Group':<24s} {'FGS':>6s} {'Surv':>6s} {'Exit':>6s} {'Eat+':>6s} {'Eat-':>6s}")
    print("-" * 56)
    for r in all_results:
        n = 5
        fgs = np.mean(r["fgs"][-n:])
        surv = np.mean(r["survival"][-n:])
        ext = np.mean(r["exit_rate"][-n:])
        eg = np.mean(r["eat_beneficial"][-n:])
        eb = np.mean(r["eat_harmful"][-n:])
        print(f"{r['name']:<24s} {fgs:>+5.3f} {surv:>5.0%} {ext:>5.0%} {eg:>5.0%} {eb:>5.0%}")

    save_dir = os.path.dirname(os.path.abspath(__file__)) or "."
    plot_results(all_results, save_dir)
    print("\nDone!")

if __name__ == "__main__":
    main()
