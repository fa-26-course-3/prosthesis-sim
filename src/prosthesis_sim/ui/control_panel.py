"""tkinter control panel: pick **mode** (human / prosthesis / full arm), then a
sensory scenario. Clicking a scenario button launches a MuJoCo passive viewer
in a background thread that runs the chosen env + sensory pipeline + controller.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Callable, Optional

import mujoco
import mujoco.viewer

from prosthesis_sim.controllers import (
    BaseController, PDMuscleController, ScriptedController, MotorNetPolicyController,
    WebcamTeleopController,
)
from prosthesis_sim.envs import MyoArmConfig, MyoArmEnv, MODES, get_spec
from prosthesis_sim.envs.arm_specs import TARGET_POSES
from prosthesis_sim.sensory import SensoryPipeline, preset


MODE_LABELS = {
    "human_elbow":     "Human — biological elbow (6-muscle, 2-joint)",
    "prosthesis_arm":  "Prosthesis — full MPL arm (MyoChallenge-24 bionic hand)",
}


@dataclass
class ScenarioButton:
    label: str
    description: str
    sensory_preset: str
    controller_kind: str
    add_ball: bool = False
    add_button: bool = False


def _scenarios() -> list[ScenarioButton]:
    return [
        ScenarioButton("1. Open-loop (arm_heben demo)",
                       "Scripted flex/extend cycle, no feedback.",
                       "healthy", "scripted", add_ball=True),
        ScenarioButton("2. PD, full proprioception",
                       "Closed-loop PD with perfect joint feedback.",
                       "healthy", "pd_muscle"),
        ScenarioButton("3. PD, deafferented",
                       "No proprioception (Jayasinghe et al.).",
                       "deafferented", "pd_muscle"),
        ScenarioButton("4. PD, noisy proprio",
                       "Spindle/GTO noise + 20 ms delay.",
                       "noisy_proprio", "pd_muscle"),
        ScenarioButton("5. PD, delayed proprio",
                       "50 ms transmission delay.",
                       "delayed_proprio", "pd_muscle"),
        ScenarioButton("6. Prosthesis (learned RNN)",
                       "Visual-only + MotorNet-style GRU policy.",
                       "visual_only", "motornet"),
        ScenarioButton("7. Press the button (prosthesis only)",
                       "MyoChallenge-style manipulation: reach + press a red button.",
                       "healthy", "pd_muscle", add_button=True),
        ScenarioButton("8. Webcam teleop (prosthesis only)",
                       "Track your real right hand via webcam and mirror it onto the MPL.",
                       "healthy", "webcam_teleop"),
        ScenarioButton("9. Webcam teleop + button",
                       "Same but a red button is present so you can press it live.",
                       "healthy", "webcam_teleop", add_button=True),
        ScenarioButton("10. Dual-view teleop (camera + prosthesis)",
                       "Left: annotated webcam with shoulder/elbow/wrist markers, "
                       "lines and angles. Right: MuJoCo viewer mimicking your arm.",
                       "healthy", "dual_view_teleop"),
    ]


# ----------------------------------------------------------------- runtime --
class SimRunner:
    def __init__(self, mode: str, scenario: ScenarioButton, target_pose: str,
                 log_cb: Callable[[str], None]):
        self.mode = mode
        self.scenario = scenario
        self.target_pose = target_pose
        self._log = log_cb
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            self._log("[warn] a simulation is already running; stop it first.")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # Kill a dual-view subprocess if one is running.
        proc = getattr(self, "_subproc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def wait(self, timeout: float = 3.0) -> None:
        """Block until the background thread finishes, so MuJoCo memory is freed
        before the next scenario is built."""
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run_dual_view_subprocess(self, s) -> None:
        """Launch scripts/run_teleop_dual_view.py in a fresh process.

        Isolates cv2's UI event loop and the MuJoCo viewer from tkinter's own
        mainloop -- otherwise all three fight for the Windows message queue.
        """
        script = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "scripts",
            "run_teleop_dual_view.py",
        )
        script = os.path.abspath(script)
        args = [sys.executable, script,
                "--sensory", s.sensory_preset,
                "--target-fps", "30"]
        if s.add_button:
            args.append("--with-button")
        try:
            self._subproc = subprocess.Popen(args, cwd=os.path.dirname(script) + "/..")
            self._log(f"[dual-view] launched PID {self._subproc.pid}")
            while not self._stop.is_set() and self._subproc.poll() is None:
                time.sleep(0.2)
            if self._subproc.poll() is None:
                self._subproc.terminate()
                self._subproc.wait(timeout=3.0)
            self._log(f"[dual-view] exited with code {self._subproc.returncode}")
        except Exception as e:
            self._log(f"[dual-view] failed: {e!r}")

    def _build_ctrl(self, env) -> BaseController:
        kind = self.scenario.controller_kind
        if kind == "webcam_teleop":
            return WebcamTeleopController.from_env(env)
        if kind == "dual_view_teleop":
            return WebcamTeleopController.from_env(
                env, tracker_kind="full",
                tracker_kwargs={"target_fps": 30.0, "draw_overlay": True},
            )
        if kind == "scripted":
            return ScriptedController.from_env(env, hold_steps=500)
        if kind == "pd_muscle":
            return PDMuscleController.from_env(env)
        if kind == "motornet":
            return MotorNetPolicyController.from_env(env)
        raise ValueError(f"unknown controller kind {kind!r}")

    def _run(self) -> None:
        s = self.scenario
        # Only prosthesis_arm honours target-pose selection; other modes ignore.
        use_pose = self.target_pose if self.mode == "prosthesis_arm" else None
        self._log(
            f"[run] mode={self.mode}  scenario='{s.label}'  sensory={s.sensory_preset}"
            + (f"  target_pose={use_pose}" if use_pose else "")
            + ("  (button task)" if s.add_button else "")
        )
        # Dual-view teleop is run in a subprocess so cv2.imshow and the MuJoCo
        # viewer share a fresh event loop, isolated from tkinter's mainloop.
        if s.controller_kind == "dual_view_teleop":
            self._run_dual_view_subprocess(s)
            return
        env = None
        ctrl = None
        try:
            env = MyoArmEnv(MyoArmConfig(
                mode=self.mode, add_ball=s.add_ball, add_button=s.add_button,
                target_pose=use_pose, horizon_s=20.0,
            ))
            pipe = SensoryPipeline(preset(s.sensory_preset), dt=env.cfg.timestep,
                                   layout=env.state_layout())
            ctrl = self._build_ctrl(env); ctrl.reset()

            raw = env.reset()
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                last_press_logged = False
                while viewer.is_running() and not self._stop.is_set():
                    obs = pipe.observe(raw)
                    a = ctrl.act(obs)
                    raw, r, _, done, info = env.step(a)
                    if info.get("button_pressed") and not last_press_logged:
                        last_press_logged = True
                        step = info.get("button_press_step")
                        self._log(f"[button] PRESSED at step {step} ({step*env.cfg.timestep:.2f}s)")
                    viewer.sync()
                    time.sleep(env.cfg.timestep)
                    if done:
                        raw = env.reset(); pipe.reset(); ctrl.reset()
                        last_press_logged = False
        except Exception as e:
            self._log(f"[error] {e!r}")
        finally:
            if ctrl is not None:
                try:
                    close = getattr(ctrl, "close", None)
                    if callable(close):
                        close()
                except Exception:
                    pass
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            # Release MuJoCo native buffers before the next scenario is built.
            gc.collect()
            self._log(f"[done] scenario='{s.label}'")


# ------------------------------------------------------------------- app --
class ControlPanel(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Ferienakademie 2026 — Prosthesis Control Sandbox")
        self.geometry("820x760")
        self._runner: Optional[SimRunner] = None
        self._mode_var = tk.StringVar(value="human_elbow")
        self._pose_var = tk.StringVar(value="reach_forward")

        # ---- mode selector -------------------------------------------------
        top = ttk.LabelFrame(self, text="Arm mode")
        top.pack(fill="x", padx=12, pady=(12, 4))
        for mode in MODES:
            spec = get_spec(mode)
            ttk.Radiobutton(
                top, text=MODE_LABELS.get(mode, mode),
                value=mode, variable=self._mode_var,
                command=self._on_mode_change,
            ).pack(anchor="w", padx=8, pady=2)
        self._mode_hint = ttk.Label(self, text="", foreground="#555",
                                    wraplength=780, justify="left")
        self._mode_hint.pack(anchor="w", padx=16, pady=(0, 6))

        # ---- target-pose selector (only used by prosthesis_arm) ------------
        pose_frame = ttk.LabelFrame(self, text="Target pose (prosthesis_arm only)")
        pose_frame.pack(fill="x", padx=12, pady=4)
        for k, pose in enumerate(TARGET_POSES):
            ttk.Radiobutton(pose_frame, text=pose, value=pose, variable=self._pose_var
                           ).grid(row=k // 3, column=k % 3, sticky="w", padx=8, pady=2)

        self._on_mode_change()

        # ---- scenarios -----------------------------------------------------
        mid = ttk.LabelFrame(self, text="Sensory scenario")
        mid.pack(fill="x", padx=12, pady=4)
        for scen in _scenarios():
            row = ttk.Frame(mid); row.pack(fill="x", pady=2, padx=6)
            ttk.Button(row, text=scen.label, width=40,
                       command=lambda s=scen: self._launch(s)).pack(side="left")
            ttk.Label(row, text=scen.description, wraplength=380,
                      foreground="#666").pack(side="left", padx=8)

        # ---- utility bar ---------------------------------------------------
        util = ttk.Frame(self); util.pack(fill="x", padx=12, pady=6)
        ttk.Button(util, text="Stop current sim",
                   command=self._stop).pack(side="left")
        ttk.Button(util, text="Train prosthesis policy (~1 min CPU)",
                   command=self._train).pack(side="left", padx=8)

        # ---- log -----------------------------------------------------------
        self.log = tk.Text(self, height=14, wrap="word", state="disabled",
                           font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, padx=12, pady=8)
        self._log_line("Ready. Pick a mode, target pose, then a scenario.")

    # ------------------------------------------------------------ helpers
    def _current_mode(self) -> str:
        return self._mode_var.get()
# Signal the previous run to stop AND wait for its viewer/env to be
        # fully released before we try to compile a new heavy MuJoCo model --
        # otherwise "Could not allocate memory" fires during spec.compile().
        if self._runner is not None:
            self._runner.stop()
            self._runner.wait(timeout=3.0)
        gc.collect
    def _on_mode_change(self) -> None:
        spec = get_spec(self._current_mode())
        self._mode_hint.configure(text=f"→ {spec.description}")

    def _log_line(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", msg.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _launch(self, scen: ScenarioButton) -> None:
        self._stop()
        self._runner = SimRunner(self._current_mode(), scen,
                                 self._pose_var.get(), self._log_line)
        self._runner.start()

    def _stop(self) -> None:
        if self._runner:
            self._runner.stop()
            self._log_line("[stop] signalled current sim to stop")

    def _train(self) -> None:
        mode = self._current_mode()
        def _worker():
            self._log_line(f"[train:{mode}] starting MotorNet-style RNN training…")
            try:
                from prosthesis_sim.training.train_motornet import train
                p = train(mode=mode, episodes=40, horizon_s=1.5)
                self._log_line(f"[train:{mode}] weights saved to {p}")
            except Exception as e:
                self._log_line(f"[train:{mode}] FAILED: {e!r}")
        threading.Thread(target=_worker, daemon=True).start()


def main() -> None:
    ControlPanel().mainloop()


if __name__ == "__main__":
    main()
