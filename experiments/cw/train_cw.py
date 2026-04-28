#!/usr/bin/env python3

# Usage:
#   Local test:
#     python experiments/cw/train_cw.py --config configs/cw_baseline.yaml --seed-index 0
#
#   Slurm run: Use run_cw_oscar.sh 
#     CONFIG=configs/cw_baseline.yaml sbatch --array=0-0 experiments/cw/run_cw_oscar.sh
#
# Outputs:
#   experiments/cw/results/<experiment>/<agent>/seed_<id>/
#     model_final.zip
#     metrics.json
#     tb_logs/


from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.cw_adapter import make_cw_env, make_cw_sequence  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CW tasks with optional HACE reward wrapper.")
    p.add_argument("--config", type=str, default="configs/cw_baseline.yaml")
    p.add_argument("--pilot", action="store_true", help="Use pilot overrides from config.")
    p.add_argument("--seed-index", type=int, default=None, help="Run only this seed index.")
    p.add_argument("--output-dir", type=str, default=None, help="Override logging.save_dir.")
    p.add_argument("--agent", type=str, default=None, help="Run one agent by name.")
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("pyyaml is required to load configs. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _extract_success(info: Dict[str, Any]) -> Optional[float]:
    for k in ("cw_success", "success", "is_success", "episode_success", "task_success"):
        if k in info:
            try:
                return float(info[k])
            except Exception:
                return 1.0 if bool(info[k]) else 0.0
    return None

def log_progress(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)
def make_progress_callback(enabled: bool, label: str, total_steps: int, interval: int):
    from stable_baselines3.common.callbacks import BaseCallback

    class ProgressCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.start_step = 0
            self.start_time = 0.0
            self.next_print = interval

        def _on_training_start(self):
            self.start_step = int(self.model.num_timesteps)
            self.start_time = time.time()

        def _on_step(self):
            if not enabled:
                return True

            done = int(self.model.num_timesteps) - self.start_step
            if done >= self.next_print:
                elapsed = time.time() - self.start_time
                pct = 100 * done / max(total_steps, 1)
                print(
                    f"[progress] {label} {done}/{total_steps} "
                    f"({pct:.0f}%) elapsed={elapsed/60:.1f}m",
                    flush=True,
                )
                self.next_print += interval

            return True

    return ProgressCallback()
# def evaluate_sb3(model, env, episodes: int, seed: int, *, deterministic: bool = True) -> Dict[str, Any]:
#     returns = []
#     lengths = []
#     successes = []
#     for ep in range(episodes):
#         obs, info = env.reset(seed=seed + ep)
#         done = False
#         ep_ret = 0.0
#         ep_len = 0
#         final_info = dict(info or {})
#         while not done:
#             action, _ = model.predict(obs, deterministic=deterministic)
#             obs, reward, terminated, truncated, info = env.step(action)
#             ep_ret += float(reward)
#             ep_len += 1
#             final_info = dict(info or {})
#             done = bool(terminated) or bool(truncated)
#         returns.append(ep_ret)
#         lengths.append(ep_len)
#         s = _extract_success(final_info)
#         if s is not None:
#             successes.append(float(s))
#     out: Dict[str, Any] = {
#         "n_episodes": int(episodes),
#         "return_mean": float(np.mean(returns)) if returns else None,
#         "return_std": float(np.std(returns)) if returns else None,
#         "length_mean": float(np.mean(lengths)) if lengths else None,
#         "length_std": float(np.std(lengths)) if lengths else None,
#     }
#     if successes:
#         out["success_mean"] = float(np.mean(successes))
#         out["success_std"] = float(np.std(successes))
#         out["solved_rate"] = float(np.mean([1.0 if s >= 0.99 else 0.0 for s in successes]))
#     else:
#         out["success_mean"] = None
#         out["success_std"] = None
#         out["solved_rate"] = None
#     return out

def evaluate_sb3(model, env, episodes: int, seed: int, *, deterministic: bool = True) -> Dict[str, Any]:
    returns = []
    lengths = []
    successes = []
    success_steps = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0

        # Track whether success ever happens during the episode.
        ep_success = 0.0
        first_success_step = None

        s0 = _extract_success(dict(info or {}))
        if s0 is not None:
            ep_success = max(ep_success, float(s0))
            if float(s0) >= 0.99:
                first_success_step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_ret += float(reward)
            ep_len += 1

            s = _extract_success(dict(info or {}))
            if s is not None:
                ep_success = max(ep_success, float(s))
                if float(s) >= 0.99 and first_success_step is None:
                    first_success_step = ep_len

            done = bool(terminated) or bool(truncated)

        returns.append(ep_ret)
        lengths.append(ep_len)
        successes.append(ep_success)
        if first_success_step is not None:
            success_steps.append(first_success_step)

    out: Dict[str, Any] = {
        "n_episodes": int(episodes),
        "return_mean": float(np.mean(returns)) if returns else None,
        "return_std": float(np.std(returns)) if returns else None,
        "length_mean": float(np.mean(lengths)) if lengths else None,
        "length_std": float(np.std(lengths)) if lengths else None,
        "success_mean": float(np.mean(successes)) if successes else None,
        "success_std": float(np.std(successes)) if successes else None,
        "solved_rate": float(np.mean([1.0 if s >= 0.99 else 0.0 for s in successes])) if successes else None,
        "success_step_mean": float(np.mean(success_steps)) if success_steps else None,
        "success_step_std": float(np.std(success_steps)) if success_steps else None,
    }

    return out


def _select_agents(cfg: Dict[str, Any], selected: Optional[str]) -> list[Dict[str, Any]]:
    agents = cfg.get("agents") or [{"name": "hace", "alpha": 1.0, "beta": 1.0}]
    if selected is None:
        run_agents = cfg.get("run_agents")
        if isinstance(run_agents, str) and run_agents.strip():
            selected = run_agents.strip()
    if selected is not None:
        agents = [a for a in agents if str(a.get("name", "")) == str(selected)]
        if not agents:
            valid = ", ".join(sorted({str(a.get("name", "")) for a in (cfg.get("agents") or [])}))
            raise ValueError(f"No agent named '{selected}' in config.agents. Valid: {valid}")
    return agents


def _make_model(algo: str, policy: str, env, seed: int, ppo_cfg: Dict[str, Any], tb_log: Optional[str]):
    algo_l = str(algo).lower()
    device = str(ppo_cfg.get("device", "auto"))
    if algo_l == "ppo":
        from stable_baselines3 import PPO  # type: ignore
        return PPO(
            policy=policy,
            env=env,
            verbose=int(ppo_cfg.get("verbose", 0)),
            seed=seed,
            learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
            n_steps=int(ppo_cfg.get("n_steps", 2048)),
            batch_size=int(ppo_cfg.get("batch_size", 64)),
            n_epochs=int(ppo_cfg.get("n_epochs", 10)),
            ent_coef=float(ppo_cfg.get("ent_coef", 0.0)),
            gamma=float(ppo_cfg.get("gamma", 0.99)),
            gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
            tensorboard_log=tb_log,
            device=device,
        )
    if algo_l == "sac":
        from stable_baselines3 import SAC  # type: ignore

        return SAC(
            policy=policy,
            env=env,
            verbose=int(ppo_cfg.get("verbose", 0)),
            seed=seed,
            learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
            batch_size=int(ppo_cfg.get("batch_size", 256)),
            buffer_size=int(ppo_cfg.get("buffer_size", 1_000_000)),
            gamma=float(ppo_cfg.get("gamma", 0.99)),
            tensorboard_log=tb_log,
            device=device
        )
    raise ValueError(f"Unsupported ppo_config.algo '{algo}'. Use ppo or sac.")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    total_steps = int(cfg.get("total_steps", 1_000_000))
    eval_episodes = int(cfg.get("eval", {}).get("eval_episodes", 10))
    eval_deterministic = bool(cfg.get("eval", {}).get("deterministic", True))
    if args.pilot and "pilot" in cfg:
        total_steps = int(cfg["pilot"].get("total_steps", total_steps))
        eval_episodes = int(cfg["pilot"].get("eval_episodes", eval_episodes))
        eval_deterministic = bool(cfg["pilot"].get("eval_deterministic", eval_deterministic))

    num_seeds = int(cfg.get("num_seeds", 1))
    seed_indices = [int(args.seed_index)] if args.seed_index is not None else list(range(num_seeds))

    agents = _select_agents(cfg, args.agent)
    ppo_cfg = cfg.get("ppo_config", {}) or {}
    algo = str(ppo_cfg.get("algo", "ppo"))
    policy = str(ppo_cfg.get("policy", "MlpPolicy"))
    progress_prints = bool(cfg.get("logging", {}).get("progress_prints", True))
    progress_interval = int(cfg.get("logging", {}).get("progress_interval", 50_000))
    learn_progress_bar = bool(ppo_cfg.get("progress_bar", False))

    base_out = Path(cfg.get("logging", {}).get("save_dir", "experiments/cw/results"))
    if args.output_dir:
        base_out = Path(args.output_dir)
    exp_name = str(cfg.get("experiment", "cw_hace"))
    base_out = base_out / exp_name
    base_out.mkdir(parents=True, exist_ok=True)

    env_cfg = cfg.get("env", {}) or {}
    single_task = env_cfg.get("task_name")
    seq_cfg = cfg.get("sequence", {}) or {}
    seq_name = seq_cfg.get("name")
    seq_tasks = list(seq_cfg.get("tasks", []) or [])
    if (single_task is None) == (not seq_tasks):
        raise ValueError("Set exactly one of env.task_name or sequence.tasks in config.")

    if seq_tasks:
        seq_spec = make_cw_sequence(str(seq_name or "custom"), seq_tasks, steps_per_task=int(cfg.get("steps_per_task", total_steps // len(seq_tasks))))
        steps_per_task = int(cfg.get("steps_per_task", seq_spec.steps_per_task))
    else:
        seq_spec = None
        steps_per_task = None

    for agent in agents:
        name = str(agent.get("name", "agent"))
        alpha = float(agent.get("alpha", 1.0))
        beta = float(agent.get("beta", 1.0))
        for seed_idx in seed_indices:
            seed = int(1000 + seed_idx * 17)
            log_progress(
                progress_prints,
                f"[start] agent={name} seed_index={seed_idx} seed={seed} alpha={alpha} beta={beta}",
            )
            run_dir = base_out / name / f"seed_{seed_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)

            tb_log = str(run_dir / "tb_logs") if bool(cfg.get("logging", {}).get("tensorboard", True)) else None
            start = time.time()

            if single_task is not None:
                env = make_cw_env(task_name=str(single_task), seed=seed, hace=True, hace_alpha=alpha, hace_beta=beta)
                model = _make_model(algo, policy, env, seed, ppo_cfg, tb_log)
                log_progress(
                    progress_prints,
                    f"[train] agent={name} seed_index={seed_idx} task={single_task} steps={total_steps}",
                )

                model.learn(
                    total_timesteps=int(total_steps),
                    progress_bar=learn_progress_bar,
                    callback=make_progress_callback(
                    progress_prints,
                    f"agent={name} seed={seed_idx} task={single_task}",
                    int(total_steps),
                    progress_interval,
                ),
                )

                log_progress(
                    progress_prints,
                    f"[done] agent={name} seed_index={seed_idx} task={single_task}",
                )
                eval_env = make_cw_env(task_name=str(single_task), seed=seed + 1_000_000, hace=True, hace_alpha=alpha, hace_beta=beta)
                eval_payload = {
                    "single": evaluate_sb3(
                        model,
                        eval_env,
                        episodes=int(eval_episodes),
                        seed=seed + 2_000_000,
                        deterministic=eval_deterministic,
                    )
                }
                eval_env.close()
                env.close()
            else:
                assert seq_spec is not None
                env0 = make_cw_env(task_name=str(seq_spec.tasks[0]), seed=seed, hace=True, hace_alpha=alpha, hace_beta=beta)
                model = _make_model(algo, policy, env0, seed, ppo_cfg, tb_log)
                
                for task_i, task_name in enumerate(seq_spec.tasks):
                    task_start = time.time()

                    log_progress(
                        progress_prints,
                        f"[train] {name} seed={seed_idx} task={task_i + 1}/{len(seq_spec.tasks)} "
                        f"{task_name} steps={steps_per_task}",
                    )

                    env_t = make_cw_env(
                        task_name=str(task_name),
                        seed=seed + task_i * 10_000,
                        hace=True,
                        hace_alpha=alpha,
                        hace_beta=beta,
                    )
                    model.set_env(env_t)

                    model.learn(
                        total_timesteps=int(steps_per_task),
                        reset_num_timesteps=False,
                        progress_bar=learn_progress_bar,
                        callback=make_progress_callback(
                        progress_prints,
                        f"{name} seed={seed_idx} task={task_i + 1}/{len(seq_spec.tasks)} {task_name}",
                        int(steps_per_task),
                        progress_interval,
                    ),
                    )

                    env_t.close()

                    log_progress(
                        progress_prints,
                        f"[done]  {name} seed={seed_idx} task={task_i + 1}/{len(seq_spec.tasks)} "
                        f"{task_name} elapsed={(time.time() - task_start) / 60:.1f} min",
                    )
                env0.close()
                log_progress(
                    progress_prints,
                    f"[eval]  {name} seed={seed_idx} sequence={seq_spec.name}",
                )
                eval_payload = {"sequence": {"name": seq_spec.name, "tasks": {}}}
                for task_i, task_name in enumerate(seq_spec.tasks):
                    env_t = make_cw_env(task_name=str(task_name), seed=seed + 1_000_000 + task_i * 10_000, hace=True, hace_alpha=alpha, hace_beta=beta)
                    eval_payload["sequence"]["tasks"][f"task_{task_i}:{task_name}"] = evaluate_sb3(
                        model,
                        env_t,
                        episodes=int(eval_episodes),
                        seed=seed + 2_000_000 + task_i * 10_000,
                        deterministic=eval_deterministic,
                    )
                    env_t.close()

            elapsed = time.time() - start
            model_path = run_dir / "model_final"
            model.save(str(model_path))

            with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "config_path": args.config,
                        "experiment": exp_name,
                        "agent": {"name": name, "alpha": alpha, "beta": beta},
                        "seed_index": seed_idx,
                        "seed": seed,
                        "total_steps": total_steps,
                        "steps_per_task": steps_per_task,
                        "eval_episodes": eval_episodes,
                        "env": {"task_name": single_task, "sequence": seq_cfg if seq_tasks else None},
                        "ppo_config": ppo_cfg,
                        "pilot": bool(args.pilot),
                    },
                    f,
                    indent=2,
                )

            with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "training_time_s": float(elapsed),
                        "total_steps": int(total_steps),
                        "sequence": None if single_task is not None else seq_spec.name,
                        "eval": eval_payload,
                    },
                    f,
                    indent=2,
                )

            print(f"[{name} seed_index={seed_idx}] Saved model: {model_path}.zip")
            print(f"[{name} seed_index={seed_idx}] Saved metrics: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
