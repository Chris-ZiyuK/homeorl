#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Ensure repo root is importable when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.coom_adapter import make_coom_env, make_coom_sequence, COOMGymnasiumAdapter  # noqa: E402
from src.envs.coom_hace_wrapper import COOMHACEWrapper, HomeostasisConfig  # noqa: E402
from src.envs.coom_obs_wrapper import FlattenFrameStackToChannels  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train COOM from config (SB3 PPO + HACE wrapper).")
    p.add_argument("--config", type=str, default="configs/coom_experiment.yaml")
    p.add_argument("--pilot", action="store_true", help="Use the config's pilot overrides")
    p.add_argument("--seed-index", type=int, default=None, help="Run only this seed index (0-based)")
    p.add_argument("--output-dir", type=str, default=None, help="Override logging.save_dir")
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("pyyaml is required to load config files (pip install pyyaml).")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _extract_success(info: Dict[str, Any]) -> Optional[float]:
    for k in ("success", "is_success", "episode_success", "task_success"):
        if k in info:
            try:
                return float(info[k])
            except Exception:
                return 1.0 if bool(info[k]) else 0.0
    return None


def _extract_success_from_stats(env) -> Optional[float]:
    """COOM exposes success via env.get_statistics(), not info."""
    if not hasattr(env, "get_statistics"):
        return None
    try:
        stats = env.get_statistics()
    except Exception:
        return None
    if not isinstance(stats, dict):
        return None
    if "success" in stats:
        try:
            return float(stats["success"])
        except Exception:
            return None
    for k, v in stats.items():
        if isinstance(k, str) and k.endswith("/success"):
            try:
                return float(v)
            except Exception:
                return None
    return None


def _binarize_success(s: Optional[float], *, threshold: float = 0.0) -> Optional[float]:
    """Convert success/progress signals to a 0/1 success indicator.

    COOM sometimes reports success as a fractional progress value (e.g. 0.1075).
    For our metrics, treat any progress > threshold as success=1.
    """
    if s is None:
        return None
    try:
        return 1.0 if float(s) > float(threshold) else 0.0
    except Exception:
        return None


def _episode_any_success(final_info: Dict[str, Any], fallback_success_value: Optional[float]) -> Optional[float]:
    """Binary success for an episode: 1 iff agent succeeded at least once in episode."""
    if "coom_any_success" in final_info:
        try:
            return 1.0 if bool(final_info["coom_any_success"]) else 0.0
        except Exception:
            pass
    # Fall back to stats/info-derived success value (may be normalized progress).
    return _binarize_success(fallback_success_value, threshold=0.0)


def evaluate_sb3(model, env, episodes: int, seed: int) -> Dict[str, Any]:
    returns = []
    lengths = []
    succs = []
    movement = []
    switches = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        final_info = dict(info or {})

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            final_info = dict(info or {})
            done = bool(terminated) or bool(truncated)

        stats = None
        if hasattr(env, "get_statistics"):
            try:
                stats = env.get_statistics()
            except Exception:
                stats = None

        if isinstance(stats, dict):
            # COOM commonly reports stats with leading slashes, e.g. "/success".
            if "/movement" in stats:
                try:
                    movement.append(float(stats["/movement"]))
                except Exception:
                    pass
            if "/switches_pressed" in stats:
                try:
                    switches.append(float(stats["/switches_pressed"]))
                except Exception:
                    pass

        returns.append(ep_ret)
        lengths.append(ep_len)
        s_raw = _extract_success(final_info)
        if s_raw is None:
            s_raw = _extract_success_from_stats(env)
        s = _episode_any_success(final_info, s_raw)
        if s is not None:
            succs.append(s)

    payload: Dict[str, Any] = {
        "n_episodes": int(episodes),
        "return_mean": float(np.mean(returns)) if returns else None,
        "return_std": float(np.std(returns)) if returns else None,
        "length_mean": float(np.mean(lengths)) if lengths else None,
        "length_std": float(np.std(lengths)) if lengths else None,
    }
    if movement:
        payload["movement_mean"] = float(np.mean(movement))
        payload["movement_std"] = float(np.std(movement))
    if switches:
        payload["switches_pressed_mean"] = float(np.mean(switches))
        payload["switches_pressed_std"] = float(np.std(switches))
    if succs:
        payload["success_mean"] = float(np.mean(succs))
        payload["success_std"] = float(np.std(succs))
        payload["success_key_detected"] = True
    else:
        payload["success_mean"] = None
        payload["success_std"] = None
        payload["success_key_detected"] = False
    return payload


def _safe_get(d: Dict[str, Any], key: str) -> Optional[float]:
    if key not in d:
        return None
    try:
        return float(d[key])
    except Exception:
        return None


def _safe_get_stats(info: Dict[str, Any]) -> Dict[str, Any]:
    stats = info.get("coom_stats", None)
    return stats if isinstance(stats, dict) else {}


def _make_vec_coom_env(
    *,
    n_envs: int,
    use_subproc: bool,
    env_id: Optional[str],
    scenario: Optional[str],
    base_seed: int,
    hace_alpha: float,
    hace_beta: float,
    health_setpoint: float,
    stamina_setpoint: float,
    stamina_source: str,
):
    """Return a Gymnasium Env, or a VecEnv when n_envs > 1 (higher rollout throughput)."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor  # type: ignore
    from stable_baselines3.common.monitor import Monitor  # type: ignore

    def _factory(rank: int) -> Callable[[], Any]:
        def _init() -> Any:
            env = make_coom_env(
                env_id=env_id,
                scenario=scenario,
                seed=int(base_seed) + int(rank) * 10_000,
                hace=True,
                hace_alpha=hace_alpha,
                hace_beta=hace_beta,
                health_setpoint=health_setpoint,
                stamina_setpoint=stamina_setpoint,
                stamina_source=stamina_source,
            )
            # Ensure SB3 can compute episode stats (rollout/ep_rew_mean, ep_len_mean).
            return Monitor(env)

        return _init

    if n_envs < 1:
        raise ValueError("ppo_config.n_envs must be >= 1")
    if n_envs == 1:
        return _factory(0)()

    factories: List[Callable[[], Any]] = [_factory(i) for i in range(n_envs)]
    if use_subproc:
        venv = SubprocVecEnv(factories)
    else:
        venv = DummyVecEnv(factories)
    return VecMonitor(venv)


def _make_episode_stats_callback(print_every: int = 1):
    """Print COOM episode stats when Monitor reports an episode end."""
    from stable_baselines3.common.callbacks import BaseCallback  # type: ignore

    class _Cb(BaseCallback):
        def __init__(self):
            super().__init__()
            self._ep_count = 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            if not isinstance(infos, list):
                return True
            for info in infos:
                if not isinstance(info, dict):
                    continue
                if "episode" not in info:
                    continue

                self._ep_count += 1
                if print_every > 1 and (self._ep_count % print_every) != 0:
                    continue

                stats = _safe_get_stats(info)
                # Prefer our episode-level definition from COOMGymnasiumAdapter:
                # success=1 iff success occurred at least once in the episode.
                suc_any = _episode_any_success(info, None)
                suc_max = _safe_get(info, "coom_success_max")

                mov = _safe_get(stats, "/movement")
                sw = _safe_get(stats, "/switches_pressed")
                print(
                    f"[episode {self._ep_count}] "
                    f"success={suc_any} (success_max={suc_max}) movement={mov} switches={sw} "
                    f"timesteps={self.num_timesteps}"
                )
            return True

    return _Cb()


def _wrap_task_env_with_hace(task_env, *, alpha: float, beta: float, health_setpoint: float,
                            stamina_setpoint: float, stamina_source: str):
    """Wrap a COOM task env (from ContinualLearningEnv.tasks) with HACE shaping + gymnasium adapter."""
    wrapped = COOMHACEWrapper(
        task_env,
        HomeostasisConfig(
            alpha=float(alpha),
            beta=float(beta),
            health_setpoint=float(health_setpoint),
            stamina_setpoint=float(stamina_setpoint),
            stamina_source=str(stamina_source),
        ),
    )
    wrapped = FlattenFrameStackToChannels(wrapped)
    return COOMGymnasiumAdapter(wrapped)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    try:
        from stable_baselines3 import PPO  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "stable-baselines3 is required for train_coom.py. "
            "Install with: pip install 'stable-baselines3[extra]'"
        ) from e

    # Apply pilot overrides
    total_steps = int(cfg.get("total_steps", 1_000_000))
    eval_episodes = int(cfg.get("eval", {}).get("eval_episodes", 10))
    if args.pilot and "pilot" in cfg:
        total_steps = int(cfg["pilot"].get("total_steps", total_steps))
        eval_episodes = int(cfg["pilot"].get("eval_episodes", eval_episodes))

    num_seeds = int(cfg.get("num_seeds", 1))
    if args.seed_index is not None:
        seed_indices = [int(args.seed_index)]
    else:
        seed_indices = list(range(num_seeds))

    # Resolve env selection (single env or sequence)
    env_cfg = cfg.get("env", {}) or {}
    env_id = env_cfg.get("env_id", None)
    scenario = env_cfg.get("scenario", None)
    sequence_name = cfg.get("sequence", None)

    if sequence_name is None:
        if (env_id is None) == (scenario is None):
            raise ValueError("Config must set exactly one of env.env_id or env.scenario (unless sequence is set).")
    else:
        if env_id is not None or scenario is not None:
            raise ValueError("If `sequence` is set, do not set env.env_id or env.scenario.")

    # Resolve HACE settings
    hace_cfg = cfg.get("hace_config", {}) or {}
    default_alpha = float(hace_cfg.get("alpha", 1.0))
    default_beta = float(hace_cfg.get("beta", 1.0))
    health_setpoint = float(hace_cfg.get("health_setpoint", 100.0))
    stamina_setpoint = float(hace_cfg.get("stamina_setpoint", 100.0))
    stamina_source = str(hace_cfg.get("stamina_source", "stamina"))

    agents = cfg.get("agents", None)
    if not agents:
        # Back-compat: single implicit agent
        agents = [{"name": "hace", "alpha": default_alpha, "beta": default_beta}]

    # Resolve PPO settings
    ppo = cfg.get("ppo_config", {}) or {}
    policy = str(ppo.get("policy", "CnnPolicy"))

    # Resolve output directory
    base_out = Path(cfg.get("logging", {}).get("save_dir", "experiments/coom/results"))
    if args.output_dir is not None:
        base_out = Path(args.output_dir)
    exp_name = str(cfg.get("experiment", "coom_hace"))
    base_out = base_out / exp_name
    base_out.mkdir(parents=True, exist_ok=True)

    # If using a sequence, build task list once (COOM manages task env objects).
    cl_env = None
    tasks = None
    if sequence_name is not None:
        cl_env = make_coom_sequence(str(sequence_name))
        tasks = list(cl_env.tasks)
        if not tasks:
            raise RuntimeError(f"COOM sequence '{sequence_name}' produced no tasks.")
        steps_per_task = int(cfg.get("steps_per_task", max(1, total_steps // len(tasks))))
    else:
        steps_per_task = None

    # Train loop over agents + seeds
    for agent_spec in agents:
        agent_name = str(agent_spec.get("name", "agent"))
        alpha = float(agent_spec.get("alpha", default_alpha))
        beta = float(agent_spec.get("beta", default_beta))

        for seed_idx in seed_indices:
            seed = int(1000 + seed_idx * 17)
            run_dir = base_out / agent_name / f"seed_{seed_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)

            with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "config_path": args.config,
                        "experiment": exp_name,
                        "agent": {"name": agent_name, "alpha": alpha, "beta": beta},
                        "seed_index": seed_idx,
                        "seed": seed,
                        "total_steps": total_steps,
                        "steps_per_task": steps_per_task,
                        "eval_episodes": eval_episodes,
                        "env": {"env_id": env_id, "scenario": scenario, "sequence": sequence_name},
                        "hace_config": {
                            "health_setpoint": health_setpoint,
                            "stamina_setpoint": stamina_setpoint,
                            "stamina_source": stamina_source,
                        },
                        "ppo_config": ppo,
                        "pilot": bool(args.pilot),
                    },
                    f,
                    indent=2,
                )

            tb_dir = run_dir / "tb_logs"
            tb_log = str(tb_dir) if bool(cfg.get("logging", {}).get("tensorboard", True)) else None

            start = time.time()

            if sequence_name is None:
                n_envs = int(ppo.get("n_envs", 1))
                use_subproc = bool(ppo.get("vec_env_subproc", True))

                env = _make_vec_coom_env(
                    n_envs=n_envs,
                    use_subproc=use_subproc,
                    env_id=env_id,
                    scenario=scenario,
                    base_seed=seed,
                    hace_alpha=alpha,
                    hace_beta=beta,
                    health_setpoint=health_setpoint,
                    stamina_setpoint=stamina_setpoint,
                    stamina_source=stamina_source,
                )

                model = PPO(
                    policy=policy,
                    env=env,
                    verbose=1,
                    seed=seed,
                    learning_rate=float(ppo.get("learning_rate", 3e-4)),
                    n_steps=int(ppo.get("n_steps", 256)),
                    batch_size=int(ppo.get("batch_size", 64)),
                    n_epochs=int(ppo.get("n_epochs", 4)),
                    ent_coef=float(ppo.get("ent_coef", 0.01)),
                    clip_range=float(ppo.get("clip_range", 0.2)),
                    gamma=float(ppo.get("gamma", 0.99)),
                    gae_lambda=float(ppo.get("gae_lambda", 0.95)),
                    tensorboard_log=tb_log,
                )

                cb = _make_episode_stats_callback(print_every=int(ppo.get("print_episode_every", 10)))
                model.learn(
                    total_timesteps=int(total_steps),
                    progress_bar=True,
                    log_interval=int(ppo.get("log_interval", 50)),
                    callback=cb,
                )
                eval_env = make_coom_env(
                    env_id=env_id,
                    scenario=scenario,
                    seed=seed + 1_000_000,
                    hace=True,
                    hace_alpha=alpha,
                    hace_beta=beta,
                    health_setpoint=health_setpoint,
                    stamina_setpoint=stamina_setpoint,
                    stamina_source=stamina_source,
                )
                eval_payload = {
                    "single": evaluate_sb3(
                        model, eval_env, episodes=int(eval_episodes), seed=seed + 2_000_000
                    )
                }
                eval_env.close()
                env.close()
            else:
                assert tasks is not None
                # Build model on first task, then switch env sequentially.
                env0 = _wrap_task_env_with_hace(
                    tasks[0],
                    alpha=alpha,
                    beta=beta,
                    health_setpoint=health_setpoint,
                    stamina_setpoint=stamina_setpoint,
                    stamina_source=stamina_source,
                )
                model = PPO(
                    policy=policy,
                    env=env0,
                    verbose=1,
                    seed=seed,
                    learning_rate=float(ppo.get("learning_rate", 3e-4)),
                    n_steps=int(ppo.get("n_steps", 256)),
                    batch_size=int(ppo.get("batch_size", 64)),
                    n_epochs=int(ppo.get("n_epochs", 4)),
                    ent_coef=float(ppo.get("ent_coef", 0.01)),
                    clip_range=float(ppo.get("clip_range", 0.2)),
                    gamma=float(ppo.get("gamma", 0.99)),
                    gae_lambda=float(ppo.get("gae_lambda", 0.95)),
                    tensorboard_log=tb_log,
                )

                # Sequential training across tasks
                for task_idx, task_env in enumerate(tasks):
                    env_t = _wrap_task_env_with_hace(
                        task_env,
                        alpha=alpha,
                        beta=beta,
                        health_setpoint=health_setpoint,
                        stamina_setpoint=stamina_setpoint,
                        stamina_source=stamina_source,
                    )
                    model.set_env(env_t)
                    cb = _make_episode_stats_callback(print_every=int(ppo.get("print_episode_every", 10)))
                    model.learn(
                        total_timesteps=int(steps_per_task),
                        reset_num_timesteps=False,
                        progress_bar=True,
                        log_interval=int(ppo.get("log_interval", 50)),
                        callback=cb,
                    )
                    env_t.close()

                # Eval on each task (fresh wrappers)
                eval_payload = {"sequence": {"name": str(sequence_name), "tasks": {}}}
                for task_idx, task_env in enumerate(tasks):
                    env_t = _wrap_task_env_with_hace(
                        task_env,
                        alpha=alpha,
                        beta=beta,
                        health_setpoint=health_setpoint,
                        stamina_setpoint=stamina_setpoint,
                        stamina_source=stamina_source,
                    )
                    eval_payload["sequence"]["tasks"][f"task_{task_idx}"] = evaluate_sb3(
                        model, env_t, episodes=int(eval_episodes), seed=seed + 2_000_000 + task_idx * 10_000
                    )
                    env_t.close()

            elapsed = time.time() - start

            model_path = run_dir / "model_final"
            model.save(str(model_path))

            metrics = {
                "training_time_s": float(elapsed),
                "total_steps": int(total_steps),
                "sequence": sequence_name,
                "eval": eval_payload,
            }

            with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            print(f"[{agent_name} seed_index={seed_idx}] Saved model: {model_path}.zip")
            print(f"[{agent_name} seed_index={seed_idx}] Saved metrics: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

