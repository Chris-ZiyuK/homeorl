"""
Homeostatic Grounding — Multi-Seed Statistical Experiment
==========================================================
Run: conda activate minihack && python grounding_multiseed.py

10 seeds × 3 groups × 3000 episodes each.
Outputs: learning curves with 95% CI, Mann-Whitney U tests, final summary table.
"""

import os, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from scipy import stats as sp_stats
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
        return (torch.FloatTensor(np.array(s)), torch.LongTensor(a),
                torch.FloatTensor(r), torch.FloatTensor(np.array(s2)),
                torch.FloatTensor(d))
    def __len__(self):
        return len(self.buf)

# ═══════════ FGS Evaluation ═══════════

def evaluate(env_cls, obs_mode, reward_type, q_net, n_eval=80, seed_base=99999):
    ate_good = ate_bad = survived = exited = 0
    for i in range(n_eval):
        env = env_cls(obs_mode=obs_mode, reward_type=reward_type)
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        while not done:
            with torch.no_grad():
                action = q_net(torch.FloatTensor(obs).unsqueeze(0)).argmax(1).item()
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        ate_good  += int(env.stats["ate_beneficial"] > 0)
        ate_bad   += int(env.stats["ate_harmful"] > 0)
        survived  += int(not env.stats["energy_depleted"])
        exited    += int(env.stats["reached_exit"])

    pg = ate_good / n_eval
    pb = ate_bad / n_eval
    return {"fgs": pg - pb, "eat+": pg, "eat-": pb,
            "surv": survived / n_eval, "exit": exited / n_eval}

# ═══════════ Single Seed Training ═══════════

def train_one_seed(obs_mode, reward_type, seed, n_episodes=3000, eval_interval=100,
                   lr=5e-4, gamma=0.99, batch_size=64,
                   eps_start=1.0, eps_end=0.05, eps_decay=1500,
                   target_update=25):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = MultiObjectEnv(obs_mode=obs_mode, reward_type=reward_type)
    q = QNet(env.obs_dim, env.action_space.n)
    qt = QNet(env.obs_dim, env.action_space.n)
    qt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)
    buf = ReplayBuffer()

    log = {"fgs": [], "surv": [], "exit": [], "eat+": [], "eat-": []}

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        for t in range(env.max_steps):
            eps = eps_end + (eps_start - eps_end) * np.exp(-ep / eps_decay)
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = q(torch.FloatTensor(obs).unsqueeze(0)).argmax(1).item()
            nobs, r, term, trunc, _ = env.step(a)
            buf.push(obs, a, r, nobs, float(term or trunc))
            obs = nobs
            if len(buf) >= batch_size:
                s, ac, rr, s2, d = buf.sample(batch_size)
                qv = q(s).gather(1, ac.unsqueeze(1))
                with torch.no_grad():
                    tgt = rr + gamma * qt(s2).max(1)[0] * (1 - d)
                loss = nn.MSELoss()(qv.squeeze(), tgt)
                opt.zero_grad(); loss.backward(); opt.step()
            if term or trunc:
                break
        if ep % target_update == 0:
            qt.load_state_dict(q.state_dict())

        if (ep + 1) % eval_interval == 0:
            m = evaluate(MultiObjectEnv, obs_mode, reward_type, q)
            for k in log:
                log[k].append(m[k])

    return log

# ═══════════ Multi-Seed Runner ═══════════

N_SEEDS = 3
N_EPISODES = 1000
EVAL_INTERVAL = 100

def run_group(group_name, obs_mode, reward_type):
    print(f"\n{'='*50}")
    print(f"  {group_name}  ({N_SEEDS} seeds × {N_EPISODES} episodes)")
    print(f"{'='*50}")

    all_logs = []
    for s in range(N_SEEDS):
        seed = 100 + s * 7
        log = train_one_seed(obs_mode, reward_type, seed, N_EPISODES, EVAL_INTERVAL)
        all_logs.append(log)

        # Final metrics for this seed
        fgs_last = np.mean(log["fgs"][-5:])
        surv_last = np.mean(log["surv"][-5:])
        sys.stdout.write(f"\r  Seed {s+1:2d}/{N_SEEDS}  "
                         f"FGS={fgs_last:+.3f}  Surv={surv_last:.0%}")
        sys.stdout.flush()
    print()

    # Stack into arrays: shape (n_seeds, n_evals)
    result = {"name": group_name, "obs_mode": obs_mode, "reward_type": reward_type}
    for k in ["fgs", "surv", "exit", "eat+", "eat-"]:
        result[k] = np.array([l[k] for l in all_logs])  # (n_seeds, n_evals)
    result["eval_eps"] = list(range(EVAL_INTERVAL, N_EPISODES + 1, EVAL_INTERVAL))
    return result

# ═══════════ Statistics ═══════════

def bootstrap_ci(arr, n_boot=2000, alpha=0.05):
    """Bootstrap 95% CI for the mean."""
    rng = np.random.default_rng(42)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

def stat_tests(results):
    """Mann-Whitney U between groups on final FGS (last 5 evals, averaged per seed)."""
    print("\n" + "=" * 60)
    print("Statistical Significance Tests (Mann-Whitney U)")
    print("=" * 60)

    # Final FGS per seed
    final = {}
    for r in results:
        final[r["name"]] = r["fgs"][:, -5:].mean(axis=1)  # (n_seeds,)

    pairs = [
        ("A: Terminal-Only", "B: Energy-Aware"),
        ("A: Terminal-Only", "C: Energy+QUERY"),
        ("B: Energy-Aware",  "C: Energy+QUERY"),
        ("A: Terminal-Only", "D: HRRL-Driven"),
        ("B: Energy-Aware",  "D: HRRL-Driven"),
    ]

    for a_name, b_name in pairs:
        a_vals = final[a_name]
        b_vals = final[b_name]
        U, p = sp_stats.mannwhitneyu(a_vals, b_vals, alternative='two-sided')
        diff = b_vals.mean() - a_vals.mean()
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

        ci_a = bootstrap_ci(a_vals)
        ci_b = bootstrap_ci(b_vals)

        print(f"\n  {a_name} vs {b_name}")
        print(f"    {a_name}: M={a_vals.mean():+.3f}  95%CI=[{ci_a[0]:+.3f}, {ci_a[1]:+.3f}]")
        print(f"    {b_name}: M={b_vals.mean():+.3f}  95%CI=[{ci_b[0]:+.3f}, {ci_b[1]:+.3f}]")
        print(f"    Diff={diff:+.3f}  U={U:.0f}  p={p:.4f}  {sig}")

# ═══════════ Plotting ═══════════

def plot_with_ci(results, save_dir="."):
    colors = {"A: Terminal-Only": "#ef4444",
              "B: Energy-Aware":  "#f59e0b",
              "C: Energy+QUERY":  "#10b981",
              "D: HRRL-Driven":   "#8b5cf6"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Homeostatic Grounding ({N_SEEDS} seeds, 95% CI)",
                 fontsize=15, fontweight="bold")

    plots = [
        ("fgs",  "Functional Grounding Score (FGS)", axes[0, 0]),
        ("surv", "Survival Rate",                    axes[0, 1]),
        ("eat+", "P(eat beneficial)",                axes[1, 0]),
        ("eat-", "P(eat harmful)",                   axes[1, 1]),
    ]

    for key, title, ax in plots:
        for r in results:
            eps = r["eval_eps"]
            data = r[key]  # (n_seeds, n_evals)
            mean = data.mean(axis=0)
            lo = np.percentile(data, 2.5, axis=0)
            hi = np.percentile(data, 97.5, axis=0)

            c = colors[r["name"]]
            ax.plot(eps, mean, color=c, label=r["name"], linewidth=2.5)
            ax.fill_between(eps, lo, hi, color=c, alpha=0.15)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Training Episode")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if key == "fgs":
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylabel("FGS = P(eat good) - P(eat bad)")

    plt.tight_layout()
    path = os.path.join(save_dir, "grounding_multiseed.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: {path}")
    return path

# ═══════════ Main ═══════════

def main():
    print("=" * 60)
    print("Homeostatic Grounding — Multi-Seed Experiment")
    print(f"  {N_SEEDS} seeds × 3 groups × {N_EPISODES} episodes")
    print("=" * 60)

    groups = [
        ("A: Terminal-Only", "terminal", "terminal"),
        ("B: Energy-Aware",  "energy", "terminal"),
        ("C: Energy+QUERY",  "full", "terminal"),
        ("D: HRRL-Driven",   "energy", "hrrl"),
    ]

    all_results = []
    for name, mode, reward_type in groups:
        result = run_group(name, mode, reward_type)
        all_results.append(result)

    # Final table
    print("\n" + "=" * 60)
    print(f"Final Results ({N_SEEDS} seeds, last 5 evals mean ± std)")
    print("=" * 60)
    print(f"{'Group':<24s} {'FGS':>12s} {'Survive':>12s} {'Exit':>12s}")
    print("-" * 52)
    for r in all_results:
        for metric, label in [("fgs",""), ("surv",""), ("exit","")]:
            pass
        fgs_final = r["fgs"][:, -5:].mean(axis=1)
        surv_final = r["surv"][:, -5:].mean(axis=1)
        exit_final = r["exit"][:, -5:].mean(axis=1)
        print(f"{r['name']:<24s} "
              f"{fgs_final.mean():+.3f}±{fgs_final.std():.3f} "
              f"{surv_final.mean():.1%}±{surv_final.std():.1%} "
              f"{exit_final.mean():.1%}±{exit_final.std():.1%}")

    # Statistical tests
    stat_tests(all_results)

    # Plot with CI
    save_dir = os.path.dirname(os.path.abspath(__file__)) or "."
    plot_with_ci(all_results, save_dir)
    
    # Save raw data to JSON
    import json
    json_path = os.path.join(save_dir, "grounding_multiseed_data.json")
    json_data = []
    for r in all_results:
        group_data = {
            "name": r["name"],
            "obs_mode": r["obs_mode"],
            "reward_type": r["reward_type"],
            "eval_eps": r["eval_eps"],
            "fgs": r["fgs"].tolist(),
            "surv": r["surv"].tolist(),
            "exit": r["exit"].tolist(),
            "eat+": r["eat+"].tolist(),
            "eat-": r["eat-"].tolist()
        }
        json_data.append(group_data)
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    print(f"Data saved to JSON: {json_path}")

    print("\n✅ Done!")

if __name__ == "__main__":
    main()
