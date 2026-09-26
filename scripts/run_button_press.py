"""MyoChallenge-style manipulation demo: prosthesis reaches out and presses a button.

Places a red button in front of the MPL arm, drives the four MPL arm servos to a
named reach pose, and prints when (and whether) the palm touches the button.
Optional ``--viewer`` opens the interactive MuJoCo window.
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from prosthesis_sim.controllers import PDMuscleController  # noqa: E402
from prosthesis_sim.envs import MyoArmConfig, MyoArmEnv  # noqa: E402
from prosthesis_sim.envs.arm_specs import TARGET_POSES  # noqa: E402
from prosthesis_sim.sensory import SensoryPipeline, preset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", choices=list(TARGET_POSES), default="reach_forward",
                    help="Named arm target pose (see arm_specs.TARGET_POSES).")
    ap.add_argument("--button-xyz", nargs=3, type=float,
                    default=[0.315, -0.224, 1.35],
                    help="Button pedestal world-frame position "
                         "(default: exactly at the palm's reach_forward endpoint).")
    ap.add_argument("--sensory", default="healthy",
                    choices=["healthy", "deafferented", "visual_only",
                             "noisy_proprio", "delayed_proprio"])
    ap.add_argument("--horizon-s", type=float, default=4.0)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    env = MyoArmEnv(MyoArmConfig(
        mode="prosthesis_arm",
        add_button=True,
        button_pos_world=tuple(args.button_xyz),
        target_pose=args.pose,
        horizon_s=args.horizon_s,
        random_seed=0,
    ))
    pipe = SensoryPipeline(preset(args.sensory), dt=env.cfg.timestep,
                           layout=env.state_layout())
    ctrl = PDMuscleController.from_env(env)

    print(f"[button-press] pose={args.pose}  button@{args.button_xyz}  "
          f"sensory={args.sensory}  nu={env.action_dim()}")
    print(f"[button-press] target angles (deg): {tuple(env.target_deg().tolist())}")

    raw = env.reset()
    if args.viewer:
        import mujoco, mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            while v.is_running():
                obs = pipe.observe(raw)
                a = ctrl.act(obs)
                raw, _, _, done, info = env.step(a)
                v.sync()
                time.sleep(env.cfg.timestep)
                if done:
                    _report(info, env); return
    else:
        while True:
            obs = pipe.observe(raw)
            a = ctrl.act(obs)
            raw, _, _, done, info = env.step(a)
            if done:
                _report(info, env); return


def _report(info: dict, env: MyoArmEnv) -> None:
    palm = np.asarray(info["endpoint_pos_world"], dtype=float)
    button = np.asarray(info["button_pos_world"], dtype=float)
    dist = float(np.linalg.norm(palm - button))
    print(f"[button-press] endpoint(palm) xyz = {palm.round(3).tolist()}")
    print(f"[button-press] button xyz         = {button.round(3).tolist()}")
    print(f"[button-press] palm-to-button distance = {dist:.3f} m")
    if info["button_pressed"]:
        step = info["button_press_step"]
        print(f"[button-press] BUTTON PRESSED at step {step} "
              f"({step * env.cfg.timestep:.2f}s)")
    else:
        print("[button-press] Button NOT pressed within horizon.")


if __name__ == "__main__":
    main()
