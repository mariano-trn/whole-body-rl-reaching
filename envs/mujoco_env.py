# envs/mujoco_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

import mujoco


@dataclass
class EnvConfig:
    xml_path: str = "assets/robot.xml"

    sim_dt: float = 0.002          # 500 Hz
    control_dt: float = 0.02       # 50 Hz
    episode_seconds: float = 10.0

    # Target sampling bounds (meters)
    target_x: Tuple[float, float] = (0.2, 0.9)
    target_y: Tuple[float, float] = (-0.35, 0.35)
    target_z: Tuple[float, float] = (0.7, 1.4)

    # Success condition
    success_dist: float = 0.03
    success_hold_steps: int = 25
    max_torso_angvel: float = 6.0  # rad/s (stability limit for success)

    # Early termination persistence (control steps)
    term_persist_steps: int = 5

    # Termination thresholds
    min_pelvis_height: float = 0.70
    max_roll_pitch_deg: float = 45.0
    max_torso_angvel_term: float = 10.0  # rad/s

    # Action mapping
    action_scale_rad: float = 0.25  # offsets in radians around nominal q
    action_rate_limit: float = 0.15 # max delta per control step (in normalized action space)

    # Observation features
    include_prev_action: bool = True


class WholeBodyReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, cfg: EnvConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.data = mujoco.MjData(self.model)

        # Enforce timestep from config (keep XML aligned in your file too)
        self.model.opt.timestep = cfg.sim_dt

        # Control → sim frame skip
        self.frame_skip = int(round(cfg.control_dt / cfg.sim_dt))
        assert self.frame_skip >= 1, "control_dt must be >= sim_dt"

        self.max_steps = int(round(cfg.episode_seconds / cfg.control_dt))

        # --- IDs we rely on (names come from the XML template) ---
        self.site_target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        self.site_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

        self.body_pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.body_torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        self.geom_ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.geom_rfoot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "r_foot_geom")
        self.geom_lfoot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "l_foot_geom")
        self.foot_geom_ids = {self.geom_rfoot_id, self.geom_lfoot_id}

        assert self.site_target_id >= 0, "Missing site 'target_site' in robot.xml"
        assert self.site_ee_id >= 0, "Missing site 'end_effector' in robot.xml"
        assert self.body_pelvis_id >= 0, "Missing body 'pelvis' in robot.xml"
        assert self.body_torso_id >= 0, "Missing body 'torso' in robot.xml"
        assert self.geom_ground_id >= 0, "Missing geom 'ground' in robot.xml"
        assert self.geom_rfoot_id >= 0, "Missing geom 'r_foot_geom' in robot.xml"
        assert self.geom_lfoot_id >= 0, "Missing geom 'l_foot_geom' in robot.xml"

        # --- Action space: one action per actuator ctrl ---
        self.nu = self.model.nu
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32)

        # --- Determine which joints are actuated (for q_nominal) ---
        # For MuJoCo position actuators, each actuator usually targets a joint via model.actuator_trnid
        # actuator_trnid[a,0] holds joint id for joint actuators.
        self.actuated_joint_ids = []
        for a in range(self.nu):
            j_id = int(self.model.actuator_trnid[a, 0])
            if j_id >= 0:
                self.actuated_joint_ids.append(j_id)
            else:
                self.actuated_joint_ids.append(-1)  # fallback

        # Build nominal pose in joint space (actuated joints only)
        self.q_nominal = self._build_nominal_joint_targets()

        # Initialize target so _build_obs works before first reset()
        self.target_xyz = np.array([0.6, 0.0, 1.0], dtype=np.float64)  # safe default
        # If the site exists, place it there too
        if self.site_target_id >= 0:
            self.model.site_pos[self.site_target_id] = self.target_xyz

        # Observation space (we’ll build a fixed-size vector)
        obs0 = self._build_obs(prev_action=np.zeros(self.nu, dtype=np.float32))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs0.shape, dtype=np.float32
        )

        # Episode state
        self.rng = np.random.default_rng(0)
        self.step_count = 0
        self.prev_action = np.zeros(self.nu, dtype=np.float32)

        # Success / termination persistence counters
        self.success_hold = 0
        self.term_counts = {
            "fall": 0,
            "tilt": 0,
            "nonfoot_contact": 0,
            "angvel": 0,
            "joint_limit": 0,
        }

        # Rendering
        self._renderer = None
        if render_mode in ("human", "rgb_array"):
            self._renderer = mujoco.Renderer(self.model)

    # ------------------------
    # Core: reset() and step()
    # ------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.prev_action[:] = 0.0
        self.success_hold = 0
        for k in self.term_counts:
            self.term_counts[k] = 0

        # Reset simulation state
        mujoco.mj_resetData(self.model, self.data)

        # Set root pose (freejoint): qpos[0:3]=pos, qpos[3:7]=quat(w,x,y,z) in MuJoCo
        # Put pelvis above ground; adjust z if your robot intersects the floor.
        self.data.qpos[0:3] = np.array([0.0, 0.0, 1.05], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # identity quat
        self.data.qvel[:] = 0.0

        # Apply nominal actuator targets
        self._set_actuator_targets(self.q_nominal)

        # Sample a target and move the site
        self.target_xyz = self._sample_target()
        self.model.site_pos[self.site_target_id] = self.target_xyz

        # Forward once to update kinematics
        mujoco.mj_forward(self.model, self.data)

        # Let the robot settle into stable ground contacts before the episode starts
        # (reduces false terminations due to transient contact/penetration at reset).
        for _ in range(int(1.0 / self.cfg.sim_dt)):
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

        obs = self._build_obs(prev_action=self.prev_action)
        info = {
            "target_xyz": self.target_xyz.copy(),
            "q_nominal": self.q_nominal.copy(),
        }
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1

        # 1) sanitize + rate limit normalized action
        a = np.asarray(action, dtype=np.float32).copy()
        a = np.clip(a, -1.0, 1.0)

        # Rate limit in action space (prevents jittery PPO exploration)
        da = a - self.prev_action
        max_da = self.cfg.action_rate_limit
        da = np.clip(da, -max_da, max_da)
        a = self.prev_action + da

        # 2) map normalized action to actuator joint targets around nominal
        # q_target_actuator is in *actuator ctrl units* (radians for position actuators here)
        q_target = self.q_nominal + self.cfg.action_scale_rad * a

        # 3) apply targets (MuJoCo internal PD will compute torques)
        self._set_actuator_targets(q_target)

        # 4) simulate frame_skip steps
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # 5) build termination/success logic
        terminated, success = self._compute_termination_and_success()
        truncated = self.step_count >= self.max_steps

        # 6) reward (for now: minimal stub you can replace with your full reward terms)
        # You can start with a simple dense reaching reward to validate dynamics:
        ee_pos = self.data.site_xpos[self.site_ee_id].copy()
        dist = float(np.linalg.norm(ee_pos - self.target_xyz))
        reward = -dist

        # 7) obs + info
        obs = self._build_obs(prev_action=a)
        info: Dict[str, Any] = {
            "dist_to_target": dist,
            "success": success,
            "step": self.step_count,

            # --- debug termination ---
            "term_counts": self.term_counts.copy(),
            "success_hold": self.success_hold,
        }

        # opzionale ma utilissimo: stato base (capisci subito fall/tilt/angvel)
        pelvis_z = float(self.data.xpos[self.body_pelvis_id][2])
        R = self.data.xmat[self.body_torso_id].reshape(3, 3)
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        pitch = float(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))
        torso_cvel = self.data.cvel[self.body_torso_id].copy()
        angvel_norm = float(np.linalg.norm(torso_cvel[0:3]))

        info.update({
            "pelvis_z": pelvis_z,
            "roll_deg": float(np.degrees(roll)),
            "pitch_deg": float(np.degrees(pitch)),
            "torso_angvel": angvel_norm,
        })

        info["last_violations"] = getattr(self, "last_violations", {}).copy()

        self.prev_action = a
        return obs, reward, terminated, truncated, info

    # ------------------------
    # Helpers
    # ------------------------
    def _build_nominal_joint_targets(self) -> np.ndarray:
        """
        Returns actuator ctrl targets for a stable-ish standing pose.
        This is a starter pose; you will likely tune it.
        """
        q = np.zeros(self.nu, dtype=np.float32)

        # The template XML uses ctrlrange in radians for each position actuator.
        # We'll set a nominal that bends knees slightly and counters with ankles.

        # Utility: find actuator by name (safe even if order changes)
        def a_id(name: str) -> int:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if idx < 0:
                raise ValueError(f"Actuator not found: {name}")
            return idx

        # Spine
        q[a_id("act_spine_pitch")] = 0.0

        # Arm: relaxed forward-ish (optional)
        q[a_id("act_r_shoulder_pitch")] = 0.2
        q[a_id("act_r_shoulder_roll")] = 0.0
        q[a_id("act_r_elbow_pitch")] = 0.6

        # Legs: slight squat (more stable than fully locked)
        hip_pitch = -0.25
        knee = 0.5
        ankle_pitch = -0.25

        # Right
        q[a_id("act_r_hip_yaw")] = 0.0
        q[a_id("act_r_hip_roll")] = 0.0
        q[a_id("act_r_hip_pitch")] = hip_pitch
        q[a_id("act_r_knee")] = knee
        q[a_id("act_r_ankle_pitch")] = ankle_pitch
        q[a_id("act_r_ankle_roll")] = 0.0

        # Left
        q[a_id("act_l_hip_yaw")] = 0.0
        q[a_id("act_l_hip_roll")] = 0.0
        q[a_id("act_l_hip_pitch")] = hip_pitch
        q[a_id("act_l_knee")] = knee
        q[a_id("act_l_ankle_pitch")] = ankle_pitch
        q[a_id("act_l_ankle_roll")] = 0.0

        return q

    def _set_actuator_targets(self, ctrl: np.ndarray) -> None:
        # Enforce actuator ctrlrange limits
        ctrl = np.asarray(ctrl, dtype=np.float32)
        lo = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        hi = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.data.ctrl[:] = np.clip(ctrl, lo, hi)

    def _sample_target(self) -> np.ndarray:
        x = self.rng.uniform(*self.cfg.target_x)
        y = self.rng.uniform(*self.cfg.target_y)
        z = self.rng.uniform(*self.cfg.target_z)
        return np.array([x, y, z], dtype=np.float64)

    def _compute_termination_and_success(self) -> Tuple[bool, bool]:
        """
        Implements:
        - termination with persistence (> term_persist_steps)
        - success requires N consecutive control steps
        """
        # --- basic quantities ---
        pelvis_z = float(self.data.xpos[self.body_pelvis_id][2])

        # torso orientation: use rotation matrix to compute roll/pitch
        R = self.data.xmat[self.body_torso_id].reshape(3, 3)
        # ZYX Euler from rotation matrix (roll around x, pitch around y)
        # roll = atan2(R32, R33), pitch = -asin(R31)
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        pitch = float(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))

        # torso angular velocity (world frame) from cvel: [ang(3), lin(3)] in world coords
        # data.cvel is "6D velocity of com of each body" in world frame
        torso_cvel = self.data.cvel[self.body_torso_id].copy()
        torso_angvel = torso_cvel[0:3]
        angvel_norm = float(np.linalg.norm(torso_angvel))

        # end-effector distance
        ee_pos = self.data.site_xpos[self.site_ee_id].copy()
        dist = float(np.linalg.norm(ee_pos - self.target_xyz))

        # --- termination checks (instantaneous) ---
        fall = pelvis_z < self.cfg.min_pelvis_height
        tilt = (abs(np.degrees(roll)) > self.cfg.max_roll_pitch_deg) or (abs(np.degrees(pitch)) > self.cfg.max_roll_pitch_deg)
        angvel_bad = angvel_norm > self.cfg.max_torso_angvel_term
        joint_limit = self._any_joint_limit_violated()
        nonfoot_contact = self._any_nonfoot_ground_contact()
        self.last_violations = {
            "fall": fall,
            "tilt": tilt,
            "angvel": angvel_bad,
            "joint_limit": joint_limit,
            "nonfoot_contact": nonfoot_contact,
        }

        # --- persistence counters ---
        self._update_term_counter("fall", fall)
        self._update_term_counter("tilt", tilt)
        self._update_term_counter("angvel", angvel_bad)
        self._update_term_counter("joint_limit", joint_limit)
        self._update_term_counter("nonfoot_contact", nonfoot_contact)

        terminated = any(v > self.cfg.term_persist_steps for v in self.term_counts.values())

        # --- success hold counter ---
        stable_for_success = (angvel_norm <= self.cfg.max_torso_angvel) and (not terminated)
        if (dist < self.cfg.success_dist) and stable_for_success:
            self.success_hold += 1
        else:
            self.success_hold = 0

        success = self.success_hold >= self.cfg.success_hold_steps
        if success:
            terminated = True  # end episode on success

        return terminated, success

    def _update_term_counter(self, key: str, violated: bool) -> None:
        if violated:
            self.term_counts[key] += 1
        else:
            self.term_counts[key] = 0

    def _any_joint_limit_violated(self) -> bool:
        # qpos contains freejoint + all joints. For hinge/slide joints, MuJoCo stores positions in qpos.
        # We'll check hinge joint ranges via model.jnt_range.
        qpos = self.data.qpos
        for j in range(self.model.njnt):
            j_type = int(self.model.jnt_type[j])
            if j_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            adr = int(self.model.jnt_qposadr[j])
            lo, hi = self.model.jnt_range[j]
            val = float(qpos[adr])
            if (val < lo - 1e-3) or (val > hi + 1e-3):
                return True
        return False

    def _any_nonfoot_ground_contact(self) -> bool:
        # If ground contacts any geom that is NOT a foot geom => violation
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)

            if g1 == self.geom_ground_id or g2 == self.geom_ground_id:
                other = g2 if g1 == self.geom_ground_id else g1
                if other not in self.foot_geom_ids:
                    return True
        return False

    def _build_obs(self, prev_action: np.ndarray) -> np.ndarray:
        # Joint positions/velocities: use qpos/qvel excluding freejoint.
        # qpos layout: [root(7), joints...]
        # qvel layout: [root(6), joints...]
        qpos = self.data.qpos[7:].astype(np.float32)
        qvel = self.data.qvel[6:].astype(np.float32)

        torso_quat = self.data.xquat[self.body_torso_id].astype(np.float32)  # (w,x,y,z)
        torso_pos = self.data.xpos[self.body_torso_id].astype(np.float32)

        # Torso velocities (world)
        torso_cvel = self.data.cvel[self.body_torso_id].astype(np.float32)
        angvel_w = torso_cvel[0:3]
        linvel_w = torso_cvel[3:6]

        # Transform target into torso frame
        R = self.data.xmat[self.body_torso_id].reshape(3, 3).astype(np.float32)
        target_w = self.target_xyz.astype(np.float32)
        target_in_torso = R.T @ (target_w - torso_pos)

        ee_pos = self.data.site_xpos[self.site_ee_id].astype(np.float32)
        dist = np.linalg.norm(ee_pos - target_w).astype(np.float32)

        # Contacts: binary foot contact
        r_contact, l_contact = self._foot_contacts_binary()

        obs_parts = [
            qpos, qvel,
            torso_quat,
            linvel_w, angvel_w,
            np.array([torso_pos[2]], dtype=np.float32),
            np.array([r_contact, l_contact], dtype=np.float32),
            target_in_torso,
            np.array([dist], dtype=np.float32),
        ]
        if self.cfg.include_prev_action:
            obs_parts.append(prev_action.astype(np.float32))

        return np.concatenate(obs_parts, axis=0)

    def _foot_contacts_binary(self) -> Tuple[float, float]:
        r = 0.0
        l = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if g1 == self.geom_rfoot_id or g2 == self.geom_rfoot_id:
                if g1 == self.geom_ground_id or g2 == self.geom_ground_id:
                    r = 1.0
            if g1 == self.geom_lfoot_id or g2 == self.geom_lfoot_id:
                if g1 == self.geom_ground_id or g2 == self.geom_ground_id:
                    l = 1.0
        return r, l

    def render(self):
        if self._renderer is None:
            raise RuntimeError("render_mode not enabled. Create env with render_mode='rgb_array' or 'human'.")
        self._renderer.update_scene(self.data, camera=None)
        img = self._renderer.render()
        if self.render_mode == "human":
            # Minimal: return image; you can integrate with a window if you want.
            return img
        return img

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    # Quick sanity test: does the nominal pose stand without actions?
    cfg = EnvConfig(xml_path="../assets/robot.xml")
    env = WholeBodyReachEnv(cfg, render_mode=None)

    obs, info = env.reset(seed=0)
    print("Obs dim:", obs.shape, "Target:", info["target_xyz"])

    # Hold zero actions; the env still applies q_nominal internally
    while True:
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.nu, dtype=np.float32))
        if terminated or truncated:
            print("Ended:", {"terminated": terminated, "truncated": truncated, **info})
            break

    print("Final step:", info["step"])
    env.close()
