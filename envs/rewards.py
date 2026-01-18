# envs/rewards.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class RewardWeights:
    # Core
    w_reach: float = 1.0
    w_alive: float = 0.05

    # Regularizers (start small; enable one-by-one)
    w_posture: float = 0.0
    w_smooth: float = 0.0

    # Hold shaping (optional, later)
    w_hold: float = 0.0
    hold_dist: float = 0.05
    hold_vel_scale: float = 0.5  # how strongly to penalize ee speed near target


def compute_reward(
    *,
    dist_to_target: float,
    ee_linvel: np.ndarray,
    roll_rad: float,
    pitch_rad: float,
    action: np.ndarray,
    prev_action: np.ndarray,
    alive: bool,
    weights: RewardWeights,
) -> tuple[float, Dict[str, float]]:
    """
    Modular reward:
      - reach: dense distance shaping
      - alive: constant bonus if not terminated
      - posture: penalize roll/pitch
      - smoothness: penalize action changes
      - hold: extra reward for staying close with low end-effector speed (optional)

    Returns:
      total_reward, reward_terms (for logging)
    """
    # Reach: use exp shaping for nicer scale (stable early training)
    # Alternative is -dist_to_target; exp is bounded in [0,1].
    reach = float(np.exp(-3.0 * dist_to_target))  # 3.0 is a good starting alpha

    alive_bonus = float(alive)  # 1.0 if alive else 0.0

    posture_pen = float(roll_rad * roll_rad + pitch_rad * pitch_rad)

    da = action - prev_action
    smooth_pen = float(np.dot(da, da))

    # Hold: only when close enough; encourages "staying" not just touching
    ee_speed = float(np.linalg.norm(ee_linvel))
    hold = 0.0
    if dist_to_target < weights.hold_dist:
        # reward high when speed is low; clamp to avoid negative blow-ups
        hold = float(max(0.0, 1.0 - weights.hold_vel_scale * ee_speed))

    terms = {
        "reach": reach,
        "alive": alive_bonus,
        "posture_pen": posture_pen,
        "smooth_pen": smooth_pen,
        "hold": hold,
    }

    total = (
        weights.w_reach * reach
        + weights.w_alive * alive_bonus
        - weights.w_posture * posture_pen
        - weights.w_smooth * smooth_pen
        + weights.w_hold * hold
    )

    terms["total"] = float(total)
    return float(total), terms
