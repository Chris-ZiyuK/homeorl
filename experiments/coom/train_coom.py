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
    p.add_argument(
        "--agent",
        type=str,
        default=None,
        help=(
            "Run only a single agent from config.agents by name "
            "(e.g. vanilla|hace|pure_homeo). Overrides config.run_agents."
        ),
    )
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("pyyaml is required to load config files (pip install pyyaml).")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _extract_success(info: Dict[str, Any]) -> Optional[float]:
    for k in ("coom_success", "success", "is_success", "episode_success", "task_success"):
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


def _episode_success_norm(final_info: Dict[str, Any], stats: Optional[Dict[str, Any]], fallback: Optional[float]) -> Optional[float]:
    """Continuous success signal in [0,1] for the episode."""
    if "coom_success" in final_info:
        try:
            return float(final_info["coom_success"])
        except Exception:
            pass
    if isinstance(stats, dict):
        # COOM often stores keys with leading slashes.
        for k in ("success", "/success"):
            v = _safe_get(stats, k)
            if v is not None:
                return v
        for k, v in stats.items():
            if isinstance(k, str) and k.endswith("/success"):
                try:
                    return float(v)
                except Exception:
                    return None
    if fallback is not None:
        try:
            return float(fallback)
        except Exception:
            return None
    return None


def _predict_action(model, obs, *, deterministic: bool, state=None, episode_start=None):
    """Call model.predict() with optional recurrent state."""
    try:
        # RecurrentPPO (sb3-contrib) supports (obs, state, episode_start, deterministic)
        return model.predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)
    except TypeError:
        # Classic SB3 algorithms
        return model.predict(obs, deterministic=deterministic)


def evaluate_sb3(model, env, episodes: int, seed: int, *, deterministic: bool = False) -> Dict[str, Any]:
    returns = []
    lengths = []
    succ_norms = []
    movement = []
    switches = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        final_info = dict(info or {})
        state = None
        episode_start = True

        while not done:
            action, state = _predict_action(
                model,
                obs,
                deterministic=deterministic,
                state=state,
                episode_start=episode_start,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            final_info = dict(info or {})
            done = bool(terminated) or bool(truncated)
            episode_start = False

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
        s_norm = _episode_success_norm(final_info, stats if isinstance(stats, dict) else None, s_raw)
        if s_norm is not None:
            succ_norms.append(s_norm)

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
    if succ_norms:
        payload["success_norm_mean"] = float(np.mean(succ_norms))
        payload["success_norm_std"] = float(np.std(succ_norms))
        payload["solved_rate"] = float(np.mean([1.0 if s >= 0.99 else 0.0 for s in succ_norms]))
        payload["success_key_detected"] = True
    else:
        payload["success_norm_mean"] = None
        payload["success_norm_std"] = None
        payload["solved_rate"] = None
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
    start_method: Optional[str],
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
        # IMPORTANT: when using CUDA, prefer spawn to avoid CUDA illegal memory access
        # caused by forking after CUDA context initialization.
        kwargs = {}
        if start_method is not None:
            kwargs["start_method"] = str(start_method)
        venv = SubprocVecEnv(factories, **kwargs)
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
                suc_norm = _episode_success_norm(info, stats, None)

                mov = _safe_get(stats, "/movement")
                sw = _safe_get(stats, "/switches_pressed")
                print(
                    f"[episode {self._ep_count}] "
                    f"success_norm={suc_norm} movement={mov} switches={sw} "
                    f"timesteps={self.num_timesteps}"
                )
            return True

    return _Cb()


def _wrap_task_env_with_hace(task_env, *, alpha: float, beta: float, health_setpoint: float,
                            stamina_setpoint: float, stamina_source: str, flatten_action_space: bool = False):
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
    wrapped = COOMGymnasiumAdapter(wrapped)
    if flatten_action_space:
        try:
            import gymnasium as gymnasium
        except Exception:
            gymnasium = None
        if gymnasium is not None and isinstance(getattr(wrapped, "action_space", None), gymnasium.spaces.MultiDiscrete):
            from src.envs.action_space_wrappers import MultiDiscreteToDiscrete

            wrapped = MultiDiscreteToDiscrete(wrapped)
    return wrapped


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
    try:
        from stable_baselines3 import DQN  # type: ignore
    except Exception:
        DQN = None  # type: ignore
    try:
        from sb3_contrib import RecurrentPPO  # type: ignore
    except Exception:
        RecurrentPPO = None  # type: ignore

    # Apply pilot overrides
    total_steps = int(cfg.get("total_steps", 1_000_000))
    eval_episodes = int(cfg.get("eval", {}).get("eval_episodes", 10))
    eval_deterministic = bool(cfg.get("eval", {}).get("deterministic", False))
    if args.pilot and "pilot" in cfg:
        total_steps = int(cfg["pilot"].get("total_steps", total_steps))
        eval_episodes = int(cfg["pilot"].get("eval_episodes", eval_episodes))
        eval_deterministic = bool(cfg["pilot"].get("eval_deterministic", eval_deterministic))

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

    # Optional agent selection (decouple vanilla/hace/pure_homeo runs).
    selected = args.agent
    if selected is None:
        ra = cfg.get("run_agents", None)
        if isinstance(ra, str) and ra.strip():
            selected = ra.strip()
        elif isinstance(ra, list) and len(ra) == 1 and isinstance(ra[0], str):
            selected = str(ra[0])
    if selected is not None:
        agents = [a for a in agents if str(a.get("name", "")) == str(selected)]
        if not agents:
            raise ValueError(
                f"No agent named '{selected}' found in config.agents. "
                "Valid names are: "
                + ", ".join(sorted({str(a.get('name', '')) for a in (cfg.get('agents', []) or [])}))
            )

    # Resolve PPO settings
    ppo = cfg.get("ppo_config", {}) or {}
    algo = str(ppo.get("algo", "ppo")).lower()
    policy = str(ppo.get("policy", "CnnPolicy"))

    if algo in ("recurrentppo", "recurrent_ppo", "rppo"):
        if RecurrentPPO is None:
            raise RuntimeError(
                "You selected ppo_config.algo=recurrent_ppo but sb3-contrib is not installed. "
                "Install with: pip install sb3-contrib"
            )
        if policy == "CnnPolicy":
            # Sensible default for pixels + recurrence
            policy = "CnnLstmPolicy"
    if algo in ("dqn",):
        if DQN is None:
            raise RuntimeError(
                "You selected ppo_config.algo=dqn but stable-baselines3 DQN is unavailable. "
                "Install with: pip install 'stable-baselines3[extra]'"
            )

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
                vec_start_method = ppo.get("vec_start_method", None)
                if vec_start_method is None:
                    # Safer default on GPU nodes.
                    try:
                        import torch

                        vec_start_method = "spawn" if torch.cuda.is_available() else None
                    except Exception:
                        vec_start_method = None

                if algo in ("dqn",) and n_envs != 1:
                    raise ValueError("SB3 DQN only supports n_envs=1 in this script (off-policy). Set ppo_config.n_envs: 1")

                env = _make_vec_coom_env(
                    n_envs=n_envs,
                    use_subproc=use_subproc,
                    start_method=None if vec_start_method in (None, "null") else str(vec_start_method),
                    env_id=env_id,
                    scenario=scenario,
                    base_seed=seed,
                    hace_alpha=alpha,
                    hace_beta=beta,
                    health_setpoint=health_setpoint,
                    stamina_setpoint=stamina_setpoint,
                    stamina_source=stamina_source,
                )

                if algo in ("dqn",):
                    # DQN requires a Discrete action space. Convert COOM MultiDiscrete -> Discrete.
                    try:
                        import gymnasium as gymnasium
                    except Exception:
                        gymnasium = None
                    if gymnasium is not None and isinstance(getattr(env, "action_space", None), gymnasium.spaces.MultiDiscrete):
                        from src.envs.action_space_wrappers import MultiDiscreteToDiscrete

                        env = MultiDiscreteToDiscrete(env)

                algo_kwargs = dict(
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
                if algo in ("dqn",):
                    # Off-policy hyperparameters (defaults are SB3 defaults)
                    model = DQN(
                        policy=policy,
                        env=env,
                        verbose=1,
                        seed=seed,
                        learning_rate=float(ppo.get("learning_rate", 1e-4)),
                        buffer_size=int(ppo.get("buffer_size", 1_000_000)),
                        learning_starts=int(ppo.get("learning_starts", 50_000)),
                        batch_size=int(ppo.get("batch_size", 32)),
                        tau=float(ppo.get("tau", 1.0)),
                        gamma=float(ppo.get("gamma", 0.99)),
                        train_freq=ppo.get("train_freq", 4),
                        gradient_steps=int(ppo.get("gradient_steps", 1)),
                        target_update_interval=int(ppo.get("target_update_interval", 10_000)),
                        exploration_fraction=float(ppo.get("exploration_fraction", 0.1)),
                        exploration_final_eps=float(ppo.get("exploration_final_eps", 0.05)),
                        tensorboard_log=tb_log,
                    )
                else:
                    model = (RecurrentPPO if algo in ("recurrentppo", "recurrent_ppo", "rppo") else PPO)(**algo_kwargs)

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
                    flatten_action_space=bool(algo in ("dqn",)),
                )
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
                assert tasks is not None
                # Build model on first task, then switch env sequentially.
                env0 = _wrap_task_env_with_hace(
                    tasks[0],
                    alpha=alpha,
                    beta=beta,
                    health_setpoint=health_setpoint,
                    stamina_setpoint=stamina_setpoint,
                    stamina_source=stamina_source,
                    flatten_action_space=bool(algo in ("dqn",)),
                )
                algo_kwargs = dict(
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
                if algo in ("dqn",):
                    model = DQN(
                        policy=policy,
                        env=env0,
                        verbose=1,
                        seed=seed,
                        learning_rate=float(ppo.get("learning_rate", 1e-4)),
                        buffer_size=int(ppo.get("buffer_size", 1_000_000)),
                        learning_starts=int(ppo.get("learning_starts", 50_000)),
                        batch_size=int(ppo.get("batch_size", 32)),
                        tau=float(ppo.get("tau", 1.0)),
                        gamma=float(ppo.get("gamma", 0.99)),
                        train_freq=ppo.get("train_freq", 4),
                        gradient_steps=int(ppo.get("gradient_steps", 1)),
                        target_update_interval=int(ppo.get("target_update_interval", 10_000)),
                        exploration_fraction=float(ppo.get("exploration_fraction", 0.1)),
                        exploration_final_eps=float(ppo.get("exploration_final_eps", 0.05)),
                        tensorboard_log=tb_log,
                    )
                else:
                    model = (RecurrentPPO if algo in ("recurrentppo", "recurrent_ppo", "rppo") else PPO)(**algo_kwargs)

                # Sequential training across tasks
                for task_idx, task_env in enumerate(tasks):
                    env_t = _wrap_task_env_with_hace(
                        task_env,
                        alpha=alpha,
                        beta=beta,
                        health_setpoint=health_setpoint,
                        stamina_setpoint=stamina_setpoint,
                        stamina_source=stamina_source,
                        flatten_action_space=bool(algo in ("dqn",)),
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
                        flatten_action_space=bool(algo in ("dqn",)),
                    )
                    eval_payload["sequence"]["tasks"][f"task_{task_idx}"] = evaluate_sb3(
                        model,
                        env_t,
                        episodes=int(eval_episodes),
                        seed=seed + 2_000_000 + task_idx * 10_000,
                        deterministic=eval_deterministic,
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

