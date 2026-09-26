"""Live webcam teleoperation of the MPL prosthesis hand.

Uses MediaPipe Hands to track the user's right hand from a webcam and mirrors
the pose to the ``prosthesis_arm`` MPL simulation:

* wrist position in the frame -> shoulder flex + shoulder ab/adduction
* apparent hand size in the frame -> elbow flexion
* palm plane tilt -> wrist FLEX + UDEV servos
* per-finger MCP bend -> per-finger MCP servos

Run
---
```
.\.venv\Scripts\python.exe scripts\run_webcam_teleop.py                  # camera 0
.\.venv\Scripts\python.exe scripts\run_webcam_teleop.py --camera 1       # pick device
.\.venv\Scripts\python.exe scripts\run_webcam_teleop.py --no-webcam      # synthetic
.\.venv\Scripts\python.exe scripts\run_webcam_teleop.py --with-button    # + button task
```
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mujoco, mujoco.viewer  # noqa: E402

from prosthesis_sim.controllers import WebcamTeleopController  # noqa: E402
from prosthesis_sim.envs import MyoArmConfig, MyoArmEnv  # noqa: E402
from prosthesis_sim.sensory import SensoryPipeline, preset  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0,
                    help="cv2 VideoCapture device index (default: 0).")
    ap.add_argument("--no-webcam", action="store_true",
                    help="Use the synthetic (no-camera) tracker.")
    ap.add_argument("--target-fps", type=float, default=30.0)
    ap.add_argument("--horizon-s", type=float, default=1e6,
                    help="Simulation horizon in seconds (default: run forever).")
    ap.add_argument("--sensory", default="healthy",
                    choices=["healthy", "deafferented", "visual_only",
                             "noisy_proprio", "delayed_proprio"])
    ap.add_argument("--with-button", action="store_true",
                    help="Also spawn a red button so you can 'press' it live.")
    ap.add_argument("--headless", action="store_true",
                    help="No MuJoCo viewer -- useful for CI / smoke tests.")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Stop after N physics steps (headless smoke test).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    env = MyoArmEnv(MyoArmConfig(
        mode="prosthesis_arm",
        add_button=args.with_button,
        target_pose="reach_forward",   # sane initial pose; teleop overrides it
        horizon_s=args.horizon_s,
        random_seed=0,
    ))
    pipe = SensoryPipeline(preset(args.sensory), dt=env.cfg.timestep,
                           layout=env.state_layout())

    tracker_kind = "synthetic" if args.no_webcam else "mediapipe"
    ctrl = WebcamTeleopController.from_env(
        env,
        tracker_kind=tracker_kind,
        tracker_kwargs={"camera_index": args.camera,
                        "target_fps": args.target_fps} if not args.no_webcam
                       else {"target_fps": args.target_fps},
    )
    ctrl.reset()

    print(f"[teleop] tracker = {tracker_kind}  headless = {args.headless}"
          f"  button = {args.with_button}  sensory = {args.sensory}")
    print(f"[teleop] nu = {env.action_dim()}  MPL joints = "
          f"{len(env.pos_servo_pairs)} position servos")

    raw = env.reset()
    step_count = 0
    try:
        if args.headless:
            while True:
                obs = pipe.observe(raw)
                a = ctrl.act(obs)
                raw, _, _, done, info = env.step(a)
                step_count += 1
                if info.get("button_pressed"):
                    print(f"[teleop] BUTTON PRESSED at step {step_count} "
                          f"({step_count * env.cfg.timestep:.2f}s)")
                if done or (args.max_steps and step_count >= args.max_steps):
                    break
        else:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                while viewer.is_running():
                    obs = pipe.observe(raw)
                    a = ctrl.act(obs)
                    raw, _, _, done, info = env.step(a)
                    step_count += 1
                    if info.get("button_pressed") and (step_count % 200 == 0):
                        print("[teleop] button held pressed")
                    viewer.sync()
                    time.sleep(env.cfg.timestep)
                    if done:
                        raw = env.reset(); pipe.reset(); ctrl.mapper.reset()
    finally:
        ctrl.close()
        env.close()
    print(f"[teleop] finished after {step_count} steps "
          f"({step_count * env.cfg.timestep:.2f} s of simulated time).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
