# render_video.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from envs.mujoco_env import WholeBodyReachEnv, EnvConfig


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(root: Path, ppo_cfg: Dict[str, Any], render_mode: str = "rgb_array") -> Tuple[WholeBodyReachEnv, Monitor]:
    """
    Return (base_env, wrapped_env). base_env is used for set_stage() + render().
    """
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
    base_env = WholeBodyReachEnv(env_cfg, render_mode=render_mode)
    env = Monitor(base_env)
    return base_env, env


def set_eval_condition(base_env: WholeBodyReachEnv, cond: str) -> Dict[str, Any]:
    """
    Apply deterministic evaluation settings (no curriculum).
    Returns a dict describing the applied settings for overlay/report.
    """
    if cond == "easy":
        st = {
            "name": "Eval-Easy",
            "target_x": [0.25, 0.45],
            "target_y": [-0.08, 0.08],
            "target_z": [0.90, 1.10],
            "reward": {"w_reach": 1.0, "w_alive": 0.05, "w_posture": 0.10, "w_smooth": 0.01, "w_hold": 0.10},
            "domain_randomization": False,
        }
    elif cond == "nominal":
        st = {
            "name": "Eval-Nominal",
            "target_x": [0.20, 0.90],
            "target_y": [-0.35, 0.35],
            "target_z": [0.70, 1.40],
            "reward": {"w_reach": 1.0, "w_alive": 0.05, "w_posture": 0.20, "w_smooth": 0.01, "w_hold": 0.20},
            "domain_randomization": False,
        }
    elif cond == "robust":
        st = {
            "name": "Eval-Robust-ready",
            "target_x": [0.20, 0.90],
            "target_y": [-0.35, 0.35],
            "target_z": [0.70, 1.40],
            "reward": {"w_reach": 1.0, "w_alive": 0.05, "w_posture": 0.20, "w_smooth": 0.01, "w_hold": 0.20},
            # Flip to True once DR is implemented in env.reset()
            "domain_randomization": False,
        }
    else:
        raise ValueError(f"Unknown condition: {cond}")

    base_env.set_stage(
        target_x=st["target_x"],
        target_y=st["target_y"],
        target_z=st["target_z"],
        reward_weights=st.get("reward", {}),
    )
    return st


def _draw_overlay(
    frame_bgr: np.ndarray,
    *,
    condition_name: str,
    model_name: str,
    episode_idx: int,
    episode_count: int,
    step_idx: int,
    control_dt: float,
    info: Dict[str, Any],
) -> np.ndarray:
    """
    Draw a readable overlay with key metrics and debug signals.
    """
    h, w = frame_bgr.shape[:2]

    # semi-transparent panel
    panel_h = 170
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), thickness=-1)
    alpha = 0.55
    frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

    def put(line: str, y: int, color=(230, 230, 230), scale=0.55, thick=1):
        cv2.putText(
            frame_bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thick,
            cv2.LINE_AA,
        )

    # Pull fields (robust to missing keys)
    dist = info.get("dist_to_target", None)
    success = bool(info.get("success", False))
    pelvis_z = info.get("pelvis_z", None)
    roll = info.get("roll_deg", None)
    pitch = info.get("pitch_deg", None)
    ang = info.get("torso_angvel", None)
    target_xyz = info.get("target_xyz", None)

    # Termination hints
    last_v = info.get("last_violations", {}) or {}
    fall_v = bool(last_v.get("fall", False))
    tilt_v = bool(last_v.get("tilt", False))
    ang_v = bool(last_v.get("angvel", False))
    jl_v = bool(last_v.get("joint_limit", False))
    nfg_v = bool(last_v.get("nonfoot_contact", False))

    t_sec = step_idx * control_dt

    # Header
    put(f"{condition_name} | episode {episode_idx+1}/{episode_count}", 26, color=(255, 255, 255), scale=0.65, thick=2)
    put(f"model: {model_name}", 50, color=(210, 210, 210), scale=0.52, thick=1)

    # Main metrics
    dist_txt = f"{dist:.3f} m" if isinstance(dist, (float, int, np.floating)) else "n/a"
    put(f"t={t_sec:5.2f}s  step={step_idx:4d}  dist={dist_txt}", 80, color=(230, 230, 230), scale=0.6, thick=1)

    if success:
        put("SUCCESS", 105, color=(80, 255, 120), scale=0.8, thick=2)
    else:
        put("success: false", 105, color=(200, 200, 200), scale=0.6, thick=1)

    # Stability signals
    pz_txt = f"{pelvis_z:.3f}" if isinstance(pelvis_z, (float, int, np.floating)) else "n/a"
    r_txt = f"{roll:+.1f}" if isinstance(roll, (float, int, np.floating)) else "n/a"
    p_txt = f"{pitch:+.1f}" if isinstance(pitch, (float, int, np.floating)) else "n/a"
    a_txt = f"{ang:.2f}" if isinstance(ang, (float, int, np.floating)) else "n/a"
    put(f"pelvis_z={pz_txt}  roll={r_txt}°  pitch={p_txt}°  torso_w={a_txt} rad/s", 135, color=(210, 210, 210), scale=0.55)

    # Violations
    v_flags = []
    if fall_v: v_flags.append("fall")
    if tilt_v: v_flags.append("tilt")
    if ang_v: v_flags.append("angvel")
    if jl_v: v_flags.append("joint_limit")
    if nfg_v: v_flags.append("nonfoot_contact")
    v_txt = " | ".join(v_flags) if v_flags else "none"
    put(f"violations: {v_txt}", 160, color=(180, 180, 180), scale=0.55)

    # Target
    if isinstance(target_xyz, (list, tuple, np.ndarray)) and len(target_xyz) == 3:
        tx, ty, tz = target_xyz
        put(f"target: [{tx:+.2f}, {ty:+.2f}, {tz:+.2f}]", panel_h + 22, color=(50, 90, 255), scale=0.55)

    return frame_bgr


def _write_separator(writer: cv2.VideoWriter, size: Tuple[int, int], fps: int, seconds: float = 0.35) -> None:
    w, h = size
    n = max(1, int(seconds * fps))
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(n):
        writer.write(frame)


def render_condition_video(
    *,
    cond_key: str,
    condition_name: str,
    model: PPO,
    base_env: WholeBodyReachEnv,
    env: Monitor,
    out_path: Path,
    episodes: int,
    seed: int,
    deterministic: bool,
    fps: int,
) -> None:
    """
    Renders `episodes` episodes to a single MP4 for the given condition.
    """
    control_dt = float(base_env.cfg.control_dt)

    # Prime to get frame size
    obs, info0 = env.reset(seed=seed)
    frame_rgb = base_env.render()
    if frame_rgb is None:
        raise RuntimeError("render() returned None. Ensure env was created with render_mode='rgb_array'.")

    # OpenCV expects BGR
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    model_name = Path(getattr(model, "policy", object()).__class__.__name__).name if model is not None else "PPO"

    # Write a short title card
    title = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(title, condition_name, (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(title, f"{episodes} episodes", (40, h // 2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    for _ in range(int(0.8 * fps)):
        writer.write(title)

    # Now render episodes
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        step_idx = 0

        # Ensure target is visible in overlay even if env doesn't return it in info
        # Our env reset returns info with target_xyz, but Monitor may drop reset info in some versions.
        # We'll re-inject from base_env.
        info_overlay: Dict[str, Any] = {"target_xyz": getattr(base_env, "target_xyz", None)}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            step_idx += 1

            # Merge overlay info
            info_overlay = dict(info_overlay)
            info_overlay.update(info)
            # always keep current target
            info_overlay["target_xyz"] = getattr(base_env, "target_xyz", info_overlay.get("target_xyz"))

            frame_rgb = base_env.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr = _draw_overlay(
                frame_bgr,
                condition_name=condition_name,
                model_name=str(Path(out_path).stem),
                episode_idx=ep,
                episode_count=episodes,
                step_idx=step_idx,
                control_dt=control_dt,
                info=info_overlay,
            )
            writer.write(frame_bgr)

        # Separator between episodes
        _write_separator(writer, (w, h), fps=fps, seconds=0.35)

    writer.release()
    print(f"[video] saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_wholebody_final.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="Deterministic actions (recommended for clean videos)")
    parser.add_argument("--outdir", type=str, default="docs/videos")
    parser.add_argument("--fps", type=int, default=50, help="Video FPS (control frequency is 50Hz by default)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    ppo_cfg = load_yaml(root / "configs" / "ppo.yaml")

    set_random_seed(int(args.seed))

    # Env must be created with rgb_array rendering enabled
    base_env, env = make_env(root, ppo_cfg, render_mode="rgb_array")

    model_path = root / args.model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # SB3 will wrap non-VecEnv as needed; fine for evaluation
    model = PPO.load(str(model_path), env=env)

    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Conditions
    conditions = [
        ("easy", "Eval-Easy"),
        ("nominal", "Eval-Nominal"),
        ("robust", "Eval-Robust-ready"),
    ]

    for i, (cond_key, _) in enumerate(conditions):
        st = set_eval_condition(base_env, cond_key)
        condition_name = st["name"]

        out_path = outdir / f"{cond_key.lower()}.mp4"

        render_condition_video(
            cond_key=cond_key,
            condition_name=condition_name,
            model=model,
            base_env=base_env,
            env=env,
            out_path=out_path,
            episodes=int(args.episodes),
            seed=int(args.seed) * 1000 + i * 100,
            deterministic=bool(args.deterministic),
            fps=int(args.fps),
        )

    env.close()


if __name__ == "__main__":
    main()
