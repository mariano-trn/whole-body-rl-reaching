# train.py
from __future__ import annotations

import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from envs.mujoco_env import WholeBodyReachEnv, EnvConfig


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CurriculumManager:
    def __init__(self, curriculum_cfg: Dict[str, Any]):
        self.cfg = curriculum_cfg
        self.window = int(curriculum_cfg["window_episodes"])
        self.advance_th = float(curriculum_cfg["advance_threshold"])
        self.allow_regress = bool(curriculum_cfg.get("allow_regress", False))
        self.regress_th = float(curriculum_cfg.get("regress_threshold", 0.0))
        self.stages: List[Dict[str, Any]] = curriculum_cfg["stages"]

        self.stage_idx = 0
        self.recent_success: List[int] = []

    @property
    def stage(self) -> Dict[str, Any]:
        return self.stages[self.stage_idx]

    def record_episode(self, success: bool):
        self.recent_success.append(1 if success else 0)
        if len(self.recent_success) > self.window:
            self.recent_success.pop(0)

    def success_rate(self) -> float:
        if not self.recent_success:
            return 0.0
        return float(np.mean(self.recent_success))

    def maybe_advance(self) -> bool:
        if len(self.recent_success) < self.window:
            return False

        sr = self.success_rate()

        # advance
        if sr >= self.advance_th and self.stage_idx < len(self.stages) - 1:
            self.stage_idx += 1
            self.recent_success.clear()
            return True

        # optional regress
        if self.allow_regress and sr <= self.regress_th and self.stage_idx > 0:
            self.stage_idx -= 1
            self.recent_success.clear()
            return True

        return False


class CurriculumCallback(BaseCallback):
    """
    Reads episode success from infos and changes env stage based on moving-window success rate.
    """

    def __init__(self, env: WholeBodyReachEnv, curriculum_cfg: Dict[str, Any], verbose: int = 1):
        super().__init__(verbose)
        self.env_ref = env
        self.cm = CurriculumManager(curriculum_cfg)
        self._apply_stage()  # stage0 at start

    def _apply_stage(self):
        st = self.cm.stage
        self.env_ref.set_stage(
            target_x=st["target_x"],
            target_y=st["target_y"],
            target_z=st["target_z"],
            reward_weights=st.get("reward", {}),
        )
        if self.verbose:
            print(f"[curriculum] -> stage {self.cm.stage_idx}: {st['name']}")

    def _on_step(self) -> bool:
        # SB3 gives a list of infos for VecEnv; Monitor wraps info and sets 'episode' at end.
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not done:
                continue

            # our env puts success in info["success"]
            success = bool(info.get("success", False))
            self.cm.record_episode(success)

            # log success rate + stage
            if self.logger is not None:
                self.logger.record("curriculum/stage_idx", self.cm.stage_idx)
                self.logger.record("curriculum/success_rate", self.cm.success_rate())

            # maybe advance
            if self.cm.maybe_advance():
                self._apply_stage()

        return True

def find_latest_checkpoint(models_dir: Path, prefix: str = "ppo_wholebody") -> Path | None:
    """
    Returns the checkpoint with the highest '<steps>_steps' suffix.
    Falls back to '<prefix>_final.zip' if present and no step checkpoints found.
    """
    if not models_dir.exists():
        return None

    # Match files like: ppo_wholebody_2800000_steps.zip
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)_steps\.zip$")

    best_path = None
    best_steps = -1

    for p in models_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            steps = int(m.group(1))
            if steps > best_steps:
                best_steps = steps
                best_path = p

    if best_path is not None:
        return best_path

    final_path = models_dir / f"{prefix}_final.zip"
    if final_path.exists():
        return final_path

    return None



def main():
    root = Path(__file__).resolve().parent
    ppo_cfg = load_yaml(root / "configs" / "ppo.yaml")
    cur_cfg = load_yaml(root / "configs" / "curriculum.yaml")

    seed = int(ppo_cfg.get("seed", 0))
    set_random_seed(seed)

    # Build EnvConfig from YAML
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

    env = WholeBodyReachEnv(env_cfg, render_mode=None)
    # Wrap with Monitor so SB3 logs episode returns/len
    env = Monitor(env)

    # Build PPO
    policy_kwargs = {"net_arch": ppo_cfg["policy_kwargs"]["net_arch"]}

    models_dir = root / ppo_cfg["train"]["save_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    ckpt = find_latest_checkpoint(models_dir, prefix="ppo_wholebody")

    if ckpt is not None:
        print(f"[resume] Loading checkpoint: {ckpt.name}")
        model = PPO.load(
            str(ckpt),
            env=env,
            tensorboard_log=str(root / ppo_cfg["train"]["tb_log_dir"]),
            seed=seed,
        )
    else:
        print("[resume] No checkpoint found, starting from scratch.")
        policy_kwargs = {"net_arch": ppo_cfg["policy_kwargs"]["net_arch"]}

        model = PPO(
            policy=ppo_cfg["ppo"]["policy"],
            env=env,
            n_steps=int(ppo_cfg["ppo"]["n_steps"]),
            batch_size=int(ppo_cfg["ppo"]["batch_size"]),
            n_epochs=int(ppo_cfg["ppo"]["n_epochs"]),
            gamma=float(ppo_cfg["ppo"]["gamma"]),
            gae_lambda=float(ppo_cfg["ppo"]["gae_lambda"]),
            clip_range=float(ppo_cfg["ppo"]["clip_range"]),
            ent_coef=float(ppo_cfg["ppo"]["ent_coef"]),
            vf_coef=float(ppo_cfg["ppo"]["vf_coef"]),
            max_grad_norm=float(ppo_cfg["ppo"]["max_grad_norm"]),
            learning_rate=float(ppo_cfg["ppo"]["learning_rate"]),
            target_kl=float(ppo_cfg["ppo"]["target_kl"]),
            policy_kwargs=policy_kwargs,
            verbose=int(ppo_cfg["ppo"]["verbose"]),
            tensorboard_log=str(root / ppo_cfg["train"]["tb_log_dir"]),
            seed=seed,
        )

    save_dir = root / ppo_cfg["train"]["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=int(ppo_cfg["train"]["save_freq_steps"]),
        save_path=str(save_dir),
        name_prefix="ppo_wholebody",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Curriculum callback needs the underlying env instance (before Monitor wrap)
    # Monitor.env is the actual env
    curriculum_cb = CurriculumCallback(env.env, cur_cfg, verbose=1)

    total_timesteps = int(ppo_cfg["train"]["total_timesteps"])
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=int(ppo_cfg["train"]["log_interval"]),
        callback=[checkpoint_cb, curriculum_cb],
        progress_bar=bool(ppo_cfg["train"].get("progress_bar", True)),
    )

    # Save final model
    model.save(str(save_dir / "ppo_wholebody_final.zip"))
    env.close()


if __name__ == "__main__":
    main()
