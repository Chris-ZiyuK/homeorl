#!/usr/bin/env python3
# Usage:
# source ~/homeorl/scripts/load_homeorl_env.sh
# cd ~/homeorl
# python experiments/cw/plot_results_simple.py --results-dir experiments/cw/results/cw_baseline

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", required=True)
parser.add_argument("--out-dir", default=None)
parser.add_argument("--metrics-name", default="metrics.json")
parser.add_argument("--metrics", default="return_mean,success_mean,solved_rate,length_mean")
parser.add_argument("--agents", default=None, help="Comma-separated, e.g. vanilla,hace")
parser.add_argument("--tb-tag", default="rollout/ep_rew_mean")
parser.add_argument("--smooth", type=int, default=5)
parser.add_argument("--no-tb", action="store_true")
parser.add_argument("--no-task-boundaries", action="store_true")
args = parser.parse_args()

results_dir = Path(args.results_dir)
out_dir = Path(args.out_dir) if args.out_dir else Path("experiments/cw/plots") / results_dir.name
out_dir.mkdir(parents=True, exist_ok=True)

agents_filter = [x.strip() for x in args.agents.split(",")] if args.agents else None
metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]

metric_labels = {
    "return_mean": "Mean return",
    "success_mean": "Success rate",
    "solved_rate": "Solved rate",
    "length_mean": "Episode length",
    "rollout/ep_rew_mean": "Mean episode reward",
}

agent_order = ["vanilla", "hace", "hace_001", "hace_003", "hace_01", "pure_homeo"]

print(f"reading: {results_dir}")
print(f"writing: {out_dir}")

# -----------------------------
# Load eval metrics
# -----------------------------
rows = []

for path in sorted(results_dir.glob(f"*/seed_*/{args.metrics_name}")):
    agent = path.parents[1].name
    if agents_filter and agent not in agents_filter:
        continue

    seed = int(path.parent.name.replace("seed_", ""))

    with open(path) as f:
        data = json.load(f)

    eval_data = data.get("eval", {})

    if "single" in eval_data:
        task = "single_task"
        cfg_path = path.parent / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            task = cfg.get("env", {}).get("task_name") or task

        row = {"agent": agent, "seed": seed, "task": task}
        row.update(eval_data["single"])
        rows.append(row)

    elif "sequence" in eval_data:
        for task, vals in eval_data["sequence"].get("tasks", {}).items():
            row = {
                "agent": agent,
                "seed": seed,
                "task": str(task).split(":", 1)[-1],
            }
            row.update(vals)
            rows.append(row)

if not rows:
    raise SystemExit(f"No {args.metrics_name} found under {results_dir}")

eval_df = pd.DataFrame(rows)

if agents_filter:
    order = agents_filter
else:
    known = [a for a in agent_order if a in set(eval_df["agent"])]
    unknown = sorted(set(eval_df["agent"]) - set(known))
    order = known + unknown

eval_df["agent"] = pd.Categorical(eval_df["agent"], categories=order, ordered=True)
eval_df = eval_df.sort_values(["agent", "seed", "task"])
eval_df.to_csv(out_dir / "eval_raw.csv", index=False)

summary_cols = [c for c in ["return_mean", "success_mean", "solved_rate", "length_mean"] if c in eval_df.columns]
print("\nsummary:")
print(eval_df.groupby("agent", observed=True)[summary_cols].mean(numeric_only=True))

# -----------------------------
# Plot eval metrics
# -----------------------------
for metric in metrics:
    if metric not in eval_df.columns:
        print(f"skip {metric}: not found")
        continue

    ylabel = metric_labels.get(metric, metric)

    task_agent = (
        eval_df.groupby(["task", "agent"], observed=True)[metric]
        .mean()
        .reset_index()
        .pivot(index="task", columns="agent", values=metric)
    )

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(task_agent.index)), 5))
    task_agent.plot(kind="bar", ax=ax, width=0.8)

    ax.set_title(f"Evaluation: {ylabel}", fontsize=15, pad=10)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    labels = [t.get_text().replace("-v1", "").replace("_", " ") for t in ax.get_xticklabels()]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Agent", frameon=True)

    plt.tight_layout()
    out = out_dir / f"eval_{metric}.png"
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"saved {out}")

    per_seed = eval_df.groupby(["agent", "seed"], observed=True)[metric].mean().reset_index()
    avg = per_seed.groupby("agent", observed=True)[metric].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(avg["agent"].astype(str), avg["mean"], yerr=avg["std"].fillna(0), capsize=4)

    ax.set_title(f"Average {ylabel}", fontsize=15, pad=10)
    ax.set_xlabel("Agent", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    labels = [t.get_text().replace("_", " ") for t in ax.get_xticklabels()]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = out_dir / f"average_{metric}.png"
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"saved {out}")

# -----------------------------
# Plot TensorBoard training curve
# -----------------------------
if not args.no_tb:
    tb_rows = []

    for seed_dir in sorted(results_dir.glob("*/seed_*")):
        agent = seed_dir.parent.name
        if agents_filter and agent not in agents_filter:
            continue

        tb_root = seed_dir / "tb_logs"
        if not tb_root.exists():
            continue

        runs = [p for p in tb_root.iterdir() if p.is_dir()]
        if not runs:
            continue

        tb_dir = max(runs, key=lambda p: p.stat().st_mtime)
        ea = EventAccumulator(str(tb_dir))
        ea.Reload()

        if args.tb_tag not in ea.Tags().get("scalars", []):
            continue

        seed = int(seed_dir.name.replace("seed_", ""))
        for e in ea.Scalars(args.tb_tag):
            tb_rows.append({"agent": agent, "seed": seed, "step": e.step, "value": e.value})

    if tb_rows:
        tb_df = pd.DataFrame(tb_rows)
        tb_df.to_csv(out_dir / "training_curve.csv", index=False)

        fig, ax = plt.subplots(figsize=(12, 5.5))

        for agent, g in tb_df.groupby("agent"):
            curve = (
                g.groupby("step")["value"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("step")
            )

            y = curve["mean"]
            if args.smooth > 1:
                y = y.rolling(args.smooth, min_periods=1).mean()

            ax.plot(curve["step"], y, linewidth=2.2, label=agent)

            if curve["count"].max() > 1:
                lower = curve["mean"] - curve["std"].fillna(0)
                upper = curve["mean"] + curve["std"].fillna(0)

                if args.smooth > 1:
                    lower = lower.rolling(args.smooth, min_periods=1).mean()
                    upper = upper.rolling(args.smooth, min_periods=1).mean()

                ax.fill_between(curve["step"], lower, upper, alpha=0.18)

        tasks = list(dict.fromkeys(eval_df["task"].astype(str).tolist()))
        if len(tasks) > 1 and not args.no_task_boundaries:
            max_step = tb_df["step"].max()
            steps_per_task = max_step / len(tasks)
            ymin, ymax = ax.get_ylim()

            for i, task in enumerate(tasks):
                x = i * steps_per_task
                ax.axvline(x, linestyle="--", linewidth=1, alpha=0.35)
                ax.text(
                    x + steps_per_task * 0.03,
                    ymax,
                    task.replace("-v1", "").replace("_", " "),
                    rotation=90,
                    va="top",
                    fontsize=9,
                    alpha=0.75,
                )

            ax.set_ylim(ymin, ymax)

        ylabel = metric_labels.get(args.tb_tag, args.tb_tag)
        ax.set_title(f"Training curve: {ylabel}", fontsize=15, pad=10)
        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title="Agent", frameon=True)

        plt.tight_layout()
        out = out_dir / "training_curve.png"
        plt.savefig(out, dpi=250)
        plt.close()
        print(f"saved {out}")
    else:
        print("no tensorboard data found")

print("\ndone")