# eval.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from envs.mujoco_env import WholeBodyReachEnv, EnvConfig


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(root: Path, ppo_cfg: Dict[str, Any]) -> Tuple[WholeBodyReachEnv, Monitor]:
    """Return (base_env, wrapped_env). base_env is needed to call set_stage()."""
    env_cfg = EnvConfig(
        xml_path=str(root / "assets" / "robot.xml"),
        sim_dt=float(ppo_cfg["env"]["sim_dt"]),
        control_dt=float(ppo_cfg["env"]["control_dt"]),
        episode_seconds=float(ppo_cfg["env"]["episode_seconds"]),
        min_pelvis_height=float(ppo_cfg["env"]["min_pelvis_height"]),
        max_roll_pitch_deg=float(ppo_cfg["env"]["max_roll_pitch_deg"]),
        max_torso_angvel_term=float(ppo_cfg["env"]["max_torso_angvel_term"]),
        action_scale_rad=float(ppo_cfg["env"]["action_scale_rad"]),
        action_rate_limit=float(ppo_cfg["env"]["action_rate_limit"]),
    )
    base_env = WholeBodyReachEnv(env_cfg, render_mode=None)
    env = Monitor(base_env)
    return base_env, env


def set_eval_condition(base_env: WholeBodyReachEnv, name: str) -> Dict[str, Any]:
    """
    Apply deterministic evaluation condition (no curriculum).
    Returns a dict describing the applied settings for reporting.
    """
    # Conditions aligned to README
    if name == "easy":
        st = {
            "name": "Eval-Easy",
            "target_x": [0.25, 0.45],
            "target_y": [-0.08, 0.08],
            "target_z": [0.90, 1.10],
            # Use final-stage reward weights or minimal shaping? For evaluation it doesn't matter much;
            # reward is optional in eval; success metrics are primary.
            "reward": {"w_reach": 1.0, "w_alive": 0.05, "w_posture": 0.10, "w_smooth": 0.01, "w_hold": 0.10},
            "domain_randomization": False,
        }
    elif name == "nominal":
        st = {
            "name": "Eval-Nominal",
            "target_x": [0.20, 0.90],
            "target_y": [-0.35, 0.35],
            "target_z": [0.70, 1.40],
            "reward": {"w_reach": 1.0, "w_alive": 0.05, "w_posture": 0.20, "w_smooth": 0.01, "w_hold": 0.20},
            "domain_randomization": False,
        }
    elif name == "robust":
        st = {
            "name": "Eval-Robust-ready",
            "target_x": [0.20, 0.90],
            "target_y": [-0.35, 0.35],
            "target_z": [0.70, 1.40],
            "reward": {"w_reach": 1.0, "w_alive": 0.05, "w_posture": 0.20, "w_smooth": 0.01, "w_hold": 0.20},
            # Placeholder: set True once DR is implemented in env.reset()
            "domain_randomization": False,
        }
    else:
        raise ValueError(f"Unknown condition: {name}")

    base_env.set_stage(
        target_x=st["target_x"],
        target_y=st["target_y"],
        target_z=st["target_z"],
        reward_weights=st["reward"],
    )
    return st


def rollout_episodes(
    model: PPO,
    env: Monitor,
    n_episodes: int,
    seed: int,
    deterministic: bool,
) -> Dict[str, Any]:
    """
    Runs n_episodes and returns metrics.
    Metrics prioritize success-based evaluation (not reward-based).
    """
    successes: List[int] = []
    fall_like: List[int] = []
    ep_lens: List[int] = []
    ep_rews: List[float] = []
    time_to_success_steps: List[int] = []

    # control steps per second = 1/control_dt; we can infer from env.unwrapped.cfg.control_dt
    # Monitor wraps WholeBodyReachEnv so env.env is the base env in most SB3 versions
    base_env = env.env  # type: ignore[attr-defined]
    control_dt = float(base_env.cfg.control_dt)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_rew = 0.0
        steps = 0
        success_step = None

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_rew += float(reward)
            steps += 1

            # first time we observe success==True, store step index
            if success_step is None and bool(info.get("success", False)):
                success_step = steps

        ep_lens.append(steps)
        ep_rews.append(ep_rew)

        s = 1 if bool(info.get("success", False)) else 0
        successes.append(s)

        # "fall rate": we approximate as termination driven by fall/tilt; if not available, fallback.
        last_v = info.get("last_violations", {})
        fell = bool(last_v.get("fall", False)) or bool(last_v.get("tilt", False))
        fall_like.append(1 if fell else 0)

        if success_step is not None:
            time_to_success_steps.append(success_step)

    success_rate = float(np.mean(successes)) if successes else 0.0
    fall_rate = float(np.mean(fall_like)) if fall_like else 0.0
    ep_len_mean = float(np.mean(ep_lens)) if ep_lens else 0.0
    ep_rew_mean = float(np.mean(ep_rews)) if ep_rews else 0.0

    # Time-to-reach: only among successful episodes
    if time_to_success_steps:
        mean_tts_steps = float(np.mean(time_to_success_steps))
        mean_tts_seconds = mean_tts_steps * control_dt
    else:
        mean_tts_steps = None
        mean_tts_seconds = None

    return {
        "n_episodes": n_episodes,
        "success_rate": success_rate,
        "fall_rate": fall_rate,
        "ep_len_mean": ep_len_mean,
        "ep_rew_mean": ep_rew_mean,
        "mean_time_to_success_steps": mean_tts_steps,
        "mean_time_to_success_seconds": mean_tts_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_wholebody_final.zip")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions for evaluation")
    parser.add_argument("--outdir", type=str, default="docs")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    ppo_cfg = load_yaml(root / "configs" / "ppo.yaml")
    set_random_seed(int(args.seed))

    base_env, env = make_env(root, ppo_cfg)

    model_path = root / args.model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = PPO.load(str(model_path), env=env)

    conditions = ["easy", "nominal", "robust"]
    results: Dict[str, Any] = {
        "model": str(model_path),
        "seed": int(args.seed),
        "episodes_per_condition": int(args.episodes),
        "deterministic": bool(args.deterministic),
        "env_config": asdict(base_env.cfg),
        "conditions": {},
    }

    for cond in conditions:
        st = set_eval_condition(base_env, cond)
        metrics = rollout_episodes(
            model=model,
            env=env,
            n_episodes=int(args.episodes),
            seed=int(args.seed) * 1000 + conditions.index(cond) * 100,
            deterministic=bool(args.deterministic),
        )
        results["conditions"][st["name"]] = {
            "settings": st,
            "metrics": metrics,
        }

        print(f"\n[{st['name']}]")
        print(json.dumps(metrics, indent=2))

    # Save JSON + TXT report
    json_path = outdir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    txt_path = outdir / "results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {results['model']}\n")
        f.write(f"Episodes/cond: {results['episodes_per_condition']}\n")
        f.write(f"Deterministic: {results['deterministic']}\n\n")
        for cname, payload in results["conditions"].items():
            m = payload["metrics"]
            f.write(f"{cname}\n")
            f.write(f"  success_rate: {m['success_rate']:.3f}\n")
            f.write(f"  fall_rate:    {m['fall_rate']:.3f}\n")
            f.write(f"  ep_len_mean:  {m['ep_len_mean']:.1f}\n")
            f.write(f"  ep_rew_mean:  {m['ep_rew_mean']:.2f}\n")
            if m["mean_time_to_success_seconds"] is not None:
                f.write(f"  mean_time_to_success_s: {m['mean_time_to_success_seconds']:.3f}\n")
            else:
                f.write("  mean_time_to_success_s: n/a\n")
            f.write("\n")

    env.close()
    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
