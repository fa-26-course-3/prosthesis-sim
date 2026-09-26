"""Dual-view live teleoperation:
* left  : annotated webcam feed (shoulder/elbow/wrist markers, connecting
          lines, real-time joint angles)
* right : MuJoCo viewer showing the MPL prosthesis mimicking your arm

Both windows are driven by the *same* MediaPipe tracker so the numbers on
screen match what the simulation is doing frame-by-frame.

Camera window is created via cv2.imshow in the main thread (MuJoCo's passive
viewer plays nicely with cv2 as long as we call ``cv2.waitKey(1)`` each
tick). Press ``q`` in the camera window or close the MuJoCo viewer to stop.

Fall-back: if the camera or MediaPipe fails, the tracker downgrades to the
synthetic backend so you can still see the pipeline run.
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2                                                 # noqa: E402
import mujoco, mujoco.viewer                               # noqa: E402
import numpy as np                                         # noqa: E402

from prosthesis_sim.controllers import WebcamTeleopController  # noqa: E402
from prosthesis_sim.envs import MyoArmConfig, MyoArmEnv        # noqa: E402
from prosthesis_sim.sensory import SensoryPipeline, preset    # noqa: E402
from prosthesis_sim.teleop import make_tracker                 # noqa: E402
from prosthesis_sim.teleop.viewer import draw_body, draw_hand, draw_hud  # noqa: E402


PANEL_TITLE = "Real arm  ->  MPL prosthesis (press 'q' to quit)"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-webcam", action="store_true")
    ap.add_argument("--target-fps", type=float, default=30.0)
    ap.add_argument("--with-button", action="store_true")
    ap.add_argument("--sensory", default="healthy",
                    choices=["healthy", "deafferented", "visual_only",
                             "noisy_proprio", "delayed_proprio"])
    ap.add_argument("--panel-x", type=int, default=40,
                    help="x-pixel where the camera panel opens.")
    ap.add_argument("--panel-y", type=int, default=80)
    ap.add_argument("--panel-width", type=int, default=680,
                    help="Camera panel width (auto-scales the frame).")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Stop after N physics steps (used for smoke tests).")
    ap.add_argument("--headless", action="store_true",
                    help="No MuJoCo viewer, no cv2 window; verify the pipeline.")
    return ap.parse_args()


def _make_placeholder_panel(width: int, height: int, msg: str) -> np.ndarray:
    """Panel shown while waiting for the first real frame."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, msg, (24, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    return img


def _scale_panel(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    ratio = target_width / float(w)
    return cv2.resize(frame, (target_width, int(h * ratio)),
                      interpolation=cv2.INTER_AREA)


def main() -> int:
    args = parse_args()

    env = MyoArmEnv(MyoArmConfig(
        mode="prosthesis_arm",
        add_button=args.with_button,
        target_pose="neutral",
        horizon_s=1e6,
        random_seed=0,
    ))
    pipe = SensoryPipeline(preset(args.sensory), dt=env.cfg.timestep,
                           layout=env.state_layout())

    tracker_kind = "synthetic" if args.no_webcam else "full"
    tracker_kwargs = ({"target_fps": args.target_fps}
                      if args.no_webcam
                      else {"camera_index": args.camera,
                            "target_fps": args.target_fps,
                            "draw_overlay": True})
    ctrl = WebcamTeleopController.from_env(
        env, tracker_kind=tracker_kind, tracker_kwargs=tracker_kwargs,
    )
    ctrl.reset()

    print(f"[teleop] tracker={tracker_kind}  headless={args.headless}"
          f"  button={args.with_button}  sensory={args.sensory}  nu={env.action_dim()}")

    show_cam = not args.headless
    if show_cam:
        # Create + place the camera window while we wait for MediaPipe to warm up.
        placeholder = _make_placeholder_panel(args.panel_width,
                                              int(args.panel_width * 3 / 4),
                                              "Waiting for the first camera frame...")
        cv2.namedWindow(PANEL_TITLE, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(PANEL_TITLE, placeholder)
        cv2.moveWindow(PANEL_TITLE, args.panel_x, args.panel_y)
        cv2.waitKey(1)

    raw = env.reset()
    step_count = 0
    quit_requested = False
    try:
        if args.headless:
            while True:
                obs = pipe.observe(raw); a = ctrl.act(obs)
                raw, _, _, done, info = env.step(a)
                step_count += 1
                if info.get("button_pressed") and step_count % 200 == 0:
                    print("[teleop] button pressed")
                if args.max_steps and step_count >= args.max_steps:
                    break
                if done:
                    raw = env.reset(); pipe.reset(); ctrl.mapper.reset()
        else:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                # Ask MuJoCo to open its window to the right of the camera panel.
                # (The exact position is up to Windows -- best effort here.)
                while viewer.is_running() and not quit_requested:
                    obs = pipe.observe(raw); a = ctrl.act(obs)
                    raw, _, _, done, info = env.step(a)
                    step_count += 1

                    if show_cam and step_count % 5 == 0:
                        frame = ctrl.tracker.latest_annotated_frame()
                        if frame is None:
                            # Synthetic mode or waiting for MediaPipe -- draw
                            # an overlay-only panel using the latest poses.
                            frame = _make_placeholder_panel(
                                args.panel_width,
                                int(args.panel_width * 3 / 4),
                                "Synthetic tracker -- no camera image.",
                            ) if tracker_kind == "synthetic" else placeholder
                            body = ctrl.tracker.latest_body()
                            hand = ctrl.tracker.latest()
                            draw_body(frame, body)
                            draw_hand(frame, hand)
                            draw_hud(frame, body, hand,
                                     fps=ctrl.tracker.fps())
                        else:
                            frame = _scale_panel(frame, args.panel_width)
                        cv2.imshow(PANEL_TITLE, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), ord("Q"), 27):
                            quit_requested = True

                    viewer.sync()
                    time.sleep(env.cfg.timestep)
                    if done:
                        raw = env.reset(); pipe.reset(); ctrl.mapper.reset()
    finally:
        try:
            ctrl.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        if show_cam:
            cv2.destroyAllWindows()
    print(f"[teleop] finished after {step_count} steps "
          f"({step_count * env.cfg.timestep:.2f} s of simulated time).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
