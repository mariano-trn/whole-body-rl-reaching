=====================================================================
Whole-Body Reinforcement Learning for Mobile Reaching in Simulation
=====================================================================

1. Overview
-----------

This repository implements a whole-body control task solved via
Reinforcement Learning in physics-based simulation.

The goal is to train a policy that can:
- Maintain balance in a humanoid / legged robot
- Reach a 3D target with an end-effector
- Coordinate upper-body motion, base stabilization, and foot contacts
- Remain robust under domain randomization

The project focuses on realistic control constraints, reward design,
curriculum learning, and reproducible evaluation.

2. Task Definition: Mobile Reach & Stabilize
--------------------------------------------

The robot starts in a standing configuration.
At the beginning of each episode, a 3D target point is sampled in space.

The policy must:
1) Keep the robot upright and stable
2) Reach the target with the end-effector
3) Maintain stable foot contacts
4) Optionally reposition (micro-steps) if needed

The target is static during the episode.


2.1 Target Space

Targets are sampled within the following bounds (meters):

- Forward (x):   [0.2, 0.9]
- Lateral (y):   [-0.35, 0.35]
- Height (z):    [0.7, 1.4]

The target is represented as a small sphere with radius 2-4 cm.


2.2 Success Condition

An episode is considered successful if, for N consecutive control steps
(N = 25 at 50 Hz):

- Distance end-effector -> target < 3 cm
- The robot has not fallen
- Torso angular velocity remains within stability limits


3. Simulation Setup
-------------------

- Physics engine: MuJoCo
- Simulation timestep: 0.002 s (500 Hz)
- Control frequency: 50 Hz
- Episode duration: max 10 seconds

Actions are applied every control step and held constant
for multiple simulation steps.


4. Robot Model
--------------

The robot is a humanoid / legged system with:
- >= 12 actuated degrees of freedom
- Two feet in contact with the ground
- A defined end-effector (hand or terminal link)

The robot model is loaded from a MuJoCo XML file and uses
realistic joint limits, inertial parameters, and contact properties.


5. Termination Conditions
-------------------------

An episode is terminated early if any of the following persist
for more than 5 control steps:

- Torso or pelvis height falls below a threshold
- Absolute roll or pitch exceeds 45 degrees
- Non-foot links contact the ground
- Excessive angular velocity of the torso
- Joint limits are violated


6. Observation Space
--------------------

The observation vector includes:

A) Proprioception
- Joint positions
- Joint velocities

B) Base / torso state
- Orientation (roll, pitch, yaw or quaternion-based encoding)
- Linear velocity (local frame)
- Angular velocity (local frame)
- Base height

C) Contacts
- Binary or continuous foot contact indicators

D) Target encoding
- Target position expressed in the torso frame
- Euclidean distance to target

E) Temporal context
- Previous actions or stacked observations

All observations are normalized using running statistics.


7. Action Space
---------------

The policy outputs normalized joint position offsets:

- Action range: [-1, 1]^n
- Actions are mapped to joint position targets
- A PD controller computes joint torques

Torque saturation and rate limits are enforced.


8. Reward Function
------------------

The total reward is a weighted sum of the following terms:

- Reach reward:
  Encourages minimizing distance between end-effector and target

- Hold reward:
  Rewards maintaining proximity to the target with low end-effector velocity

- Alive bonus:
  Small constant reward for remaining upright

- Energy penalty:
  Penalizes squared joint torques

- Smoothness penalty:
  Penalizes large action changes between steps

- Posture penalty:
  Penalizes torso roll and pitch deviations

- Foot slip penalty:
  Penalizes tangential foot velocity during contact

- Impact penalty:
  Penalizes large contact force spikes

All reward weights are explicitly defined and documented in the code.


9. Curriculum Learning
----------------------

Training progresses through multiple stages:

Stage 0:
- Standing stabilization
- Very close, easy targets

Stage 1:
- Static reaching within upper-body workspace

Stage 2:
- Wider target distribution requiring coordinated motion

Stage 3:
- Full task with domain randomization

Progression is based on measured success rate.


10. Domain Randomization
-----------------------

To improve robustness, the following parameters are randomized
at the start of each episode:

- Link masses and inertias
- Ground friction coefficient
- Actuator strength
- Sensor noise (positions, velocities, orientation)
- Optional action latency

Randomization ranges are explicitly defined and reproducible.


11. Reinforcement Learning Algorithm
------------------------------------

- Algorithm: Proximal Policy Optimization (PPO)
- Policy: MLP with two hidden layers
- Observation normalization enabled
- Fixed random seeds for reproducibility

Training is logged using TensorBoard.


12. Evaluation Protocol
-----------------------

The trained policy is evaluated under three conditions:

Eval-Easy:
- Narrow target distribution
- No domain randomization

Eval-Nominal:
- Full target distribution
- No domain randomization

Eval-Robust:
- Full target distribution
- Domain randomization enabled

Metrics reported:
- Success rate
- Mean time to reach
- Fall rate
- Energy consumption
- Foot slip statistics


13. Ablation Studies
-------------------

The following ablations are included:

1) Training without domain randomization
2) Training without foot slip penalty

These studies highlight the role of robustness and contact-aware rewards.


14. Repository Structure
------------------------

.
|-- README.txt
|-- train.py
|-- eval.py
|-- render_video.py
|-- envs/
|   |-- mujoco_env.py
|   |-- rewards.py
|   |-- termination.py
|   |-- randomization.py
|-- configs/
|   |-- ppo.yaml
|   |-- curriculum.yaml
|   |-- domain_randomization.yaml
|-- assets/
|   |-- robot.xml
|-- logs/        (ignored)
|-- models/      (saved checkpoints)
|-- docs/
|   |-- results.txt
|   |-- ablations.txt
|   |-- videos/


15. Reproducibility
-------------------

All experiments can be reproduced using fixed random seeds.
Configuration files fully specify training, curriculum,
and domain randomization parameters.


16. Limitations and Future Work
-------------------------------

- No perception (vision) input is used
- No sim-to-real transfer is attempted
- The task focuses on reaching, not object manipulation

Possible extensions include:
- Vision-based target perception
- Object interaction
- Multi-task locomotion and manipulation


17. Summary
-----------

This project demonstrates how reinforcement learning can be applied
to whole-body robotic control with realistic constraints.

The emphasis is on:
- Stable training
- Reward engineering
- Robustness
- Clear evaluation protocols

The resulting policy exhibits coordinated balance, reaching behavior,
and robustness to variations in dynamics.

=====================================================================
