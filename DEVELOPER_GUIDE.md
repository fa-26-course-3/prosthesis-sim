# Developer Guide

Single, authoritative reference for setting up, running, and extending the
sandbox end-to-end. This file is the pushed replacement for the deep-dive
docs that used to live under `docs/` (now kept only as local, git-ignored
notes under `_docs/` &mdash; they are not part of the shared repo).

Read order: this file top-to-bottom once, then use it as a reference.
For git workflow / commit conventions see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 1. Environment setup

Requirements: **Python 3.10** (MyoSuite/gymnasium currently do not support
3.11+ cleanly), Windows PowerShell (commands below), a working MuJoCo
install (installed automatically via `pip`).

```powershell
# Create and activate the virtual environment
py -3.10 -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
.\.venv\Scripts\Activate.ps1

# Install dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Verify the install end-to-end
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

`requirements.txt` pins the floor for: `mujoco`, `myosuite`, `gymnasium`,
`torch` + `motornet` (policy training), `opencv-python` + `mediapipe`
(webcam teleop), plus `numpy`/`pyyaml`/`matplotlib`/`tqdm`.

`pyproject.toml` makes `src/` importable as the `prosthesis_sim` package
(editable install via the `src` layout — no `pip install -e .` needed for
scripts, since `scripts/_bootstrap.py` adds `src/` to `sys.path`).

---

## 2. Repository layout

```
Project_Ferienakadamie_26/
├── .venv/                          # Python 3.10 virtualenv (not committed)
├── requirements.txt                # pinned dependency floor
├── pyproject.toml                  # makes src/ layout importable as a package
├── README.md                       # top-level overview / quick start
├── DEVELOPER_GUIDE.md              # this file
├── CONTRIBUTING.md                 # git/commit/branching conventions
├── configs/
│   └── default.yaml                # named scenarios consumed by utils.load_config
├── models/
│   ├── hand_landmarker.task        # MediaPipe hand tracking model
│   └── pose_landmarker.task        # MediaPipe body tracking model
├── src/prosthesis_sim/
│   ├── envs/
│   │   ├── arm_specs.py            # ArmSpec dataclass + MODES/TARGET_POSES registry
│   │   └── myo_arm_env.py          # MuJoCo/MyoSuite wrapper (reset/step/raw_state)
│   ├── sensory/
│   │   └── feedback.py             # SensoryConfig, SensoryPipeline, preset()
│   ├── controllers/
│   │   ├── base.py                 # BaseController protocol
│   │   ├── scripted.py             # arm_heben-style open loop
│   │   ├── pd_muscle.py            # joint-PD -> muscle activations / MPL servo targets
│   │   ├── motornet_policy.py      # MotorNet-style GRU policy controller
│   │   └── webcam_teleop.py        # maps live webcam pose -> MPL joint targets
│   ├── teleop/
│   │   ├── hand_pose.py            # 21 hand landmarks + finger-flex features
│   │   ├── body_pose.py            # 33 body landmarks + shoulder/elbow/wrist angles
│   │   ├── webcam_source.py        # MediaPipe HandLandmarker/PoseLandmarker trackers
│   │   ├── pose_mapper.py          # HandPose/BodyPose -> {joint_name: target_rad}
│   │   └── viewer.py               # cv2 drawing helpers (markers, lines, HUD)
│   ├── training/
│   │   └── train_motornet.py       # REINFORCE-style rollout trainer
│   ├── ui/
│   │   └── control_panel.py        # tkinter buttons -> scenarios
│   └── utils/
│       └── config.py               # YAML -> Scenario dataclasses
├── scripts/
│   ├── _bootstrap.py               # adds src/ to sys.path for direct-run scripts
│   ├── run_open_loop.py            # arm_heben demo through the modular stack
│   ├── run_feedback_ablation.py    # sweeps scenarios, prints reach-error table
│   ├── compare_modes.py            # head-to-head human vs prosthesis
│   ├── run_button_press.py         # MyoChallenge-style button manipulation demo
│   ├── run_webcam_teleop.py        # hand-only webcam teleop
│   ├── run_teleop_dual_view.py     # camera panel + MuJoCo viewer, hand+body teleop
│   ├── run_gui.py                  # launches the tkinter sandbox
│   └── smoke_test.py               # ~10 s headless end-to-end verification
└── tests/
    └── test_smoke.py               # pytest wrapper around smoke_test
```

Two folders are intentionally **not** part of the shared repo (git-ignored,
see `.gitignore`): `_docs/` (superseded internal notes/archive) and
`_documents/` (personal prompts/reference scripts). Nothing you need to run
the project lives only there — this file and the README are self-contained.

---

## 3. The core runtime loop

Every entry point (GUI button, CLI script, pytest) reduces to the same
three-object, four-step loop:

```python
from prosthesis_sim.envs import MyoArmConfig, MyoArmEnv
from prosthesis_sim.sensory import SensoryPipeline, preset
from prosthesis_sim.controllers import PDMuscleController

env  = MyoArmEnv(MyoArmConfig(mode="prosthesis_arm", horizon_s=5.0))
pipe = SensoryPipeline(preset("deafferented"), dt=env.cfg.timestep,
                       layout=env.state_layout())
ctrl = PDMuscleController.from_env(env)

raw = env.reset()
while True:
    obs = pipe.observe(raw)     # apply noise/delay/ablation
    a   = ctrl.act(obs)         # policy decides muscle/motor activations
    raw, reward, _, done, _ = env.step(a)
    if done:
        break
```

* **`raw_state`** &mdash; `[qpos, qvel, actuator_len, actuator_vel, hand_xyz]`,
  dimension depends on arm mode (19 for `human_elbow`, larger for
  `prosthesis_arm`). Slice constants live once at the top of `feedback.py`.
* **`Observation`** &mdash; a dataclass with **named** fields plus
  `proprio_available` / `vision_available` flags. Controllers never touch
  raw indices.
* **`action`** &mdash; activation/target vector, length `env.action_dim()`
  (`nu`), clipped to each actuator's valid range.

Only these three objects change between scenarios — that's what makes the
sandbox modular.

---

## 4. Arm modes in depth

| | `human_elbow` | `prosthesis_arm` |
|---|---|---|
| MyoSuite XML | `elbow/myoelbow_2dof6muscles.xml` | `envs/myo/assets/arm/myoarm_bionic_bimanual.xml` |
| DOF (`nq`) | 2 | 71 |
| Actuators (`nu`) | 6 Hill muscles | 63 Hill muscles (clamped to low tone) + 17 MPL position servos = 80 |
| Primary joints | `r_shoulder_elev`, `r_elbow_flex` | `prosthesis/Lshoulder_fe`, `Lshoulder_abad`, `Lhumeral_rot`, `Lelbow` |
| Target selection | `target_deg_override=(shoulder, elbow)` | `target_pose="reach_forward"` (or `reach_up`, `reach_across`, `reach_side`, `reach_down`, `neutral`) |
| Endpoint body | `r_ulna_radius_hand` | `prosthesis/palm` |
| Manipulation task | n/a | `add_button=True` places a red button; contact sets `info["button_pressed"]` |
| Scientific role | healthy human baseline | MyoChallenge-2024 "Prosthesis Co-Manipulation" bionic hand |

Switch modes with one field: `MyoArmConfig(mode="human_elbow")` or
`MyoArmConfig(mode="prosthesis_arm")`.

Named target poses (`arm_specs.TARGET_POSES`), `(S-flex, S-abd, H-rot, elbow)` in degrees:

| pose            | angles                | approx. palm xyz after settle |
|-----------------|------------------------|--------------------------------|
| `neutral`       | (0, 0, 0, 3)           | (0.19, 0.05, 0.87)             |
| `reach_forward` | (45, -30, 0, 90)       | (0.32, -0.22, 1.36)            |
| `reach_up`      | (120, -20, 0, 20)      | (0.37, -0.27, 1.67)            |
| `reach_across`  | (45, -90, 30, 90)      | (0.44, -0.14, 1.37)            |
| `reach_side`    | (20, -120, 0, 60)      | (0.52, -0.05, 1.61)            |
| `reach_down`    | (-30, -20, 0, 30)      | (0.34, 0.18, 0.92)             |

Registered in [`src/prosthesis_sim/envs/arm_specs.py`](src/prosthesis_sim/envs/arm_specs.py).

---

## 5. Sensory pipeline & presets

[`src/prosthesis_sim/sensory/feedback.py`](src/prosthesis_sim/sensory/feedback.py):

* `SensoryConfig` &mdash; every knob per channel (proprio, vision): on/off,
  additive Gaussian noise, per-channel delay in ms.
* `SensoryPipeline.observe(raw_state)` &mdash; slices proprio/vision blocks,
  optionally adds noise, pushes through a delay ring buffer, zeros disabled
  channels, packs everything into an `Observation`.
* `preset(name)` &mdash; scenario name -> `SensoryConfig`, shared by the GUI,
  YAML loader, and ablation script.

| preset            | proprio | proprio noise/delay | vision | vision delay | represents                          |
|-------------------|:-------:|----------------------|:------:|--------------|--------------------------------------|
| `healthy`         | on      | 0 / 0                 | on     | 100 ms       | intact upper limb                    |
| `noisy_proprio`   | on      | &sigma;=0.02 / 20 ms  | on     | 100 ms       | fatigue / illness                    |
| `delayed_proprio` | on      | 0 / 50 ms              | on     | 100 ms       | slow nerve conduction                |
| `deafferented`    | **off** | —                      | on     | 100 ms       | Jayasinghe deafferentation study      |
| `visual_only`     | **off** | —                      | on     | 120 ms       | typical prosthesis-user feedback      |

Add a new preset by adding a branch in `preset()` — every entry point
(GUI, ablation script, YAML config) picks it up automatically.

---

## 6. Controllers

All controllers implement the `Protocol` in
[`controllers/base.py`](src/prosthesis_sim/controllers/base.py):

```python
class BaseController(Protocol):
    name: str
    def reset(self) -> None: ...
    def act(self, obs: Observation, target: Optional[np.ndarray] = None) -> np.ndarray: ...
```

| file                    | class                      | role                                                                |
|-------------------------|-----------------------------|----------------------------------------------------------------------|
| `scripted.py`           | `ScriptedController`        | open-loop flex/extend cycle; reproduces the reference `arm_heben.py` |
| `pd_muscle.py`          | `PDMuscleController`        | joint-space PD; muscle agonist/antagonist split, or direct MPL servo targets; falls back to low tone when proprioception is off |
| `motornet_policy.py`    | `MotorNetPolicyController`  | small GRU sized to `env.action_dim()`; loads `models/motornet_policy_<mode>.pt` if present |
| `webcam_teleop.py`      | `WebcamTeleopController`    | reads live hand+body pose, maps to MPL joint targets via `teleop.pose_mapper` |

The factory `controllers.make(name, **kwargs)` selects one by string — used
by the YAML config loader and the GUI.

To add a new controller: create one file implementing `BaseController`,
export it in `controllers/__init__.py`, add a branch in `make()`, and
optionally add it to the GUI scenario list in `ui/control_panel.py`.

---

## 7. Sensors and real-signal integration

`MyoArmEnv.raw_state()` already exposes everything in `mjData`: joint
positions/velocities, actuator length/velocity, and endpoint xyz. These map
onto the *proprioception* (`qpos`/`qvel`/`act_len`/`act_vel`) and *vision*
(`hand_xyz`) channels degraded by `SensoryPipeline`.

Additional MuJoCo `<sensor>` types you can inject via `mujoco.MjSpec` at
runtime (same pattern as the button attachment in `MyoArmEnv._attach_button`):

| MuJoCo sensor | measures | physiological analogue |
|---|---|---|
| `jointpos` / `jointvel` | joint angle / velocity | joint proprioception |
| `actuatorfrc` | actuator force | Golgi tendon organ |
| `tendonpos` | tendon length | muscle spindle length |
| `touch` | contact force in a volume | tactile fingertip sensor |
| `force` / `torque` | 6-axis at a site | prosthesis socket load |

**Real EMG** can enter the stack in three roles, all compatible with the
existing contracts:

* **A. Direct drive** &mdash; EMG envelope directly becomes the `action`
  vector passed to `env.step()`, bypassing the controller.
* **B. Decoded intent** &mdash; a classifier turns EMG into a `target` passed
  to `ctrl.act(obs, target=...)`.
* **C. Sensory observation** &mdash; measured EMG becomes a new field on
  `Observation`, degraded by `SensoryPipeline` like any other channel.

A minimal driver would live at `src/prosthesis_sim/sensory/emg.py`
(band-pass 20–450 Hz, rectify, ~4 Hz linear envelope, normalize by MVC) and
is not shipped by default — add it if a real EMG signal is available.

**Camera / vision**: `mujoco.Renderer` renders synthetic RGB/depth frames
from the sim; the webcam teleop stack (`teleop/webcam_source.py`) shows the
pattern for wiring a real webcam via OpenCV + MediaPipe.

---

## 8. All CLI commands (reference card)

```powershell
# One-time environment
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# End-to-end smoke test (every mode x representative scenario)
.\.venv\Scripts\python.exe scripts\smoke_test.py

# Head-to-head human vs prosthesis reach-error comparison
.\.venv\Scripts\python.exe scripts\compare_modes.py

# arm_heben demo, either mode
.\.venv\Scripts\python.exe scripts\run_open_loop.py --mode human_elbow
.\.venv\Scripts\python.exe scripts\run_open_loop.py --mode prosthesis_arm

# Button-press manipulation (MyoChallenge-style)
.\.venv\Scripts\python.exe scripts\run_button_press.py --pose reach_forward --viewer

# Sensory ablation sweep across both modes
.\.venv\Scripts\python.exe scripts\run_feedback_ablation.py --all-modes

# Interactive sandbox: pick mode, target pose, click a scenario
.\.venv\Scripts\python.exe scripts\run_gui.py

# Train the MotorNet-style RNN policy per mode
.\.venv\Scripts\python.exe -m prosthesis_sim.training.train_motornet --mode prosthesis_arm

# Webcam teleop: hand-only
.\.venv\Scripts\python.exe scripts\run_webcam_teleop.py

# Webcam teleop: dual-view (camera panel + MuJoCo viewer)
.\.venv\Scripts\python.exe scripts\run_teleop_dual_view.py
.\.venv\Scripts\python.exe scripts\run_teleop_dual_view.py --with-button
.\.venv\Scripts\python.exe scripts\run_teleop_dual_view.py --sensory deafferented
.\.venv\Scripts\python.exe scripts\run_teleop_dual_view.py --no-webcam --headless --max-steps 800

# Run tests
.\.venv\Scripts\python.exe -m pytest
```

Config-driven scenarios (YAML, no code changes needed):

```python
from prosthesis_sim.utils import load_config
scenarios = load_config("configs/default.yaml")
scen = scenarios["prosthesis_arm_deafferented"]
# scen.env, scen.sensory, scen.controller, scen.controller_kwargs all ready
```

---

## 9. Where to modify what (cheat sheet)

| I want to&hellip; | Edit this file |
|---|---|
| Add a new arm model | `src/prosthesis_sim/envs/arm_specs.py` |
| Change how proprioception/vision is degraded | `src/prosthesis_sim/sensory/feedback.py` (`SensoryConfig`, `preset`) |
| Add a new controller | new file under `src/prosthesis_sim/controllers/` + update `__init__.py`/`make()` |
| Change PD gains or reach target | `PDMuscleController.__init__` kwargs, or `ArmSpec.joint_targets_deg` |
| Add a new GUI button | `_scenarios()` in `src/prosthesis_sim/ui/control_panel.py` |
| Ship a new YAML scenario | `configs/default.yaml` |
| Change webcam->MPL joint mapping | `src/prosthesis_sim/teleop/pose_mapper.py` |
| Add a real EMG source | `src/prosthesis_sim/sensory/emg.py` (new file, see Section 7) |

---

## 10. Verified smoke-test baseline

Reference output from `scripts\smoke_test.py` (re-run this after any change
that touches envs/sensory/controllers):

```
mode               ctrl         sens             nu  steps    reward  wall_s
---------------------------------------------------------------------------
human_elbow        pd_muscle    healthy           6    600   -176.92    0.05
human_elbow        pd_muscle    deafferented      6    600   -577.63    0.04
human_elbow        motornet     visual_only       6    600   -412.44    0.23
prosthesis_arm     scripted     healthy          80    600   -940.10    0.22
prosthesis_arm     pd_muscle    healthy          80    600   -332.28    0.18
prosthesis_arm     pd_muscle    deafferented     80    600   -332.28    0.21
prosthesis_arm     motornet     visual_only      80    600  -1628.62    0.33
SMOKE: PASS
```

Reading: `nu` is 6 for `human_elbow` and 80 for `prosthesis_arm`. The
prosthesis arm's `pd_muscle` reward is identical across sensory conditions
because the MPL joints are position servos with their own internal control
loop — they don't rely on biological proprioception, matching the
MyoChallenge-2024 assumption.

---

## 11. Troubleshooting / FAQ

**The MuJoCo viewer window is black on Windows.** Close and reopen — the
passive viewer occasionally starts before GLFW finishes handshaking. Not a
code issue.

**`myosuite` refuses to import.** You're on Python 3.11+/3.13. Recreate the
venv with `py -3.10 -m venv .venv` — MyoSuite currently pins a `gymnasium`
version that needs ≤3.10 wheels for some dependencies.

**`mujoco.viewer` freezes when the GUI closes.** Make sure `SimRunner.stop()`
runs (the tkinter GUI does this automatically on **Stop** or when launching a
new scenario). In scripts, close the viewer in a `finally` block.

**Where does the myoelbow XML actually live?**

```powershell
.\.venv\Scripts\python.exe -c "import os,myosuite;print(os.path.join(os.path.dirname(myosuite.__file__),'simhive/myo_sim/elbow/myoelbow_2dof6muscles.xml'))"
```

**How do I profile a controller?** Wrap the core loop (Section 3) in
`cProfile.runctx(...)` — the hot lines are always `mujoco.mj_step` and the
controller's `.act`.

---

## 12. Extending the project

* **State estimation** — add a Kalman filter after `SensoryPipeline` that
  reconstructs `qpos`/`qvel` from delayed noisy observations.
* **Inverse-dynamics feedforward** — train MotorNet's differentiable arm26
  offline for `q -> u`, use it as a bias to `PDMuscleController`.
* **Optimal control** — wrap a Bioptim `OptimalControlProgram` as a
  controller (`.act()` = one MPC horizon).
* **Real EMG drive** — see Section 7.

For git workflow, branching, and the recommended modular commit sequence for
a fresh clone, see [CONTRIBUTING.md](CONTRIBUTING.md).
