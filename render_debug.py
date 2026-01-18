import time
from pathlib import Path

import cv2
import numpy as np

from envs.mujoco_env import WholeBodyReachEnv, EnvConfig


def main():
    cfg = EnvConfig(xml_path=str(Path(__file__).resolve().parent / "assets" / "robot.xml"))
    env = WholeBodyReachEnv(cfg, render_mode="rgb_array")

    obs, info = env.reset(seed=0)
    print("Target:", info["target_xyz"])

    while True:
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.nu, dtype=np.float32))

        frame = env.render()  # RGB uint8
        # OpenCV expects BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("MuJoCo Debug View", frame_bgr)

        # ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(cfg.control_dt)

        if terminated or truncated:
            print("Ended:", {"terminated": terminated, "truncated": truncated})
            break

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
