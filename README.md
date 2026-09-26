# Ferienakademie 2026 &mdash; Course 3: Prosthesis Control Sandbox

A modular MuJoCo/MyoSuite/MotorNet sandbox that lets you simulate an upper
limb &mdash; healthy or prosthetic &mdash; and study one question:

> **How does lack of sensory feedback affect motor control, and can this be
> supplemented by (learned or hand-designed) control?**

For the full step-by-step setup, execution reference, and architecture
deep-dive, see **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)**.
For git/repo conventions, see **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## What's in the sandbox

Two arm bodies, five sensory-feedback presets, three controllers &mdash; mixed
and matched through the same three-stage loop:

```
+-----------+     raw state     +------------------+   observation   +--------------+
|  MyoArmEnv | ----------------> | SensoryPipeline  | ---------------> | Controller  |
|  (MuJoCo)  | <---------------- | (noise/delay/    | <---------------- | (the       |
|            |     action        |  ablation)       |                  |  "brain")  |
+-----------+                    +------------------+                  +--------------+
```

| arm mode         | body                                              | actuators                              | role                                   |
|------------------|-----------------------------------------------------|-----------------------------------------|-----------------------------------------|
| `human_elbow`    | biological 2-joint, 6-muscle elbow                 | 6 Hill-type muscles                     | healthy baseline                        |
| `prosthesis_arm` | myoArm + MPL bionic hand (MyoChallenge-2024 model) | 63 muscles (held at low tone) + 17 MPL servos | realistic full-arm prosthesis, incl. a button-press manipulation task |

| sensory preset      | proprioception     | vision       | represents                          |
|----------------------|:------------------:|:------------:|--------------------------------------|
| `healthy`           | on, clean          | on, 100 ms   | intact upper limb                    |
| `noisy_proprio`     | on, +noise         | on, 100 ms   | fatigue / illness                    |
| `delayed_proprio`   | on, +50 ms         | on, 100 ms   | slow nerve conduction                |
| `deafferented`      | **off**            | on, 100 ms   | Jayasinghe-style sensory loss        |
| `visual_only`       | **off**            | on, 120 ms   | typical prosthesis-user feedback     |

| controller    | what it does                                                          |
|---------------|------------------------------------------------------------------------|
| `scripted`    | open-loop flex/extend cycle (reference `arm_heben` baseline)          |
| `pd_muscle`   | closed-loop PD; drives muscles or MPL joint-angle servos               |
| `motornet`    | small GRU policy (MotorNet-style), trained via `training/train_motornet.py` |
| webcam teleop | mirrors your real arm/hand (MediaPipe) onto the MPL prosthesis        |

## Quick start

```powershell
# 1. Environment
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Headless end-to-end sanity check (both arm modes, all scenarios)
.\.venv\Scripts\python.exe scripts\smoke_test.py

# 3. Interactive sandbox with buttons (pick arm mode, target pose, scenario)
.\.venv\Scripts\python.exe scripts\run_gui.py

# 4. Head-to-head human vs. prosthesis reach-error comparison
.\.venv\Scripts\python.exe scripts\compare_modes.py

# 5. Numeric ablation sweep across sensory scenarios
.\.venv\Scripts\python.exe scripts\run_feedback_ablation.py --all-modes

# 6. Button-press manipulation demo (MyoChallenge-style)
.\.venv\Scripts\python.exe scripts\run_button_press.py --pose reach_forward --viewer

# 7. Live webcam teleoperation of the MPL prosthesis
.\.venv\Scripts\python.exe scripts\run_teleop_dual_view.py

# 8. Train the MotorNet-style RNN policy
.\.venv\Scripts\python.exe -m prosthesis_sim.training.train_motornet --mode prosthesis_arm
```

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#8-all-cli-commands-reference-card)
for the full command reference and every flag.

## Getting the code (fa-26-course-3)

This project is shared under the [fa-26-course-3](https://github.com/fa-26-course-3)
GitHub organization.

```powershell
git clone git@github.com:fa-26-course-3/<repo-name>.git
cd <repo-name>
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

If you are setting up the shared repo for the first time (initial `git init`,
remote, and the recommended modular commit sequence), follow
[`CONTRIBUTING.md`](CONTRIBUTING.md) step by step.

## Repository layout

```
├── configs/default.yaml        # named, YAML-addressable scenarios
├── models/                     # MediaPipe landmarker weights + trained policies
├── scripts/                    # runnable entry points (see quick start)
├── src/prosthesis_sim/
│   ├── envs/                   # MyoArmEnv (MuJoCo/MyoSuite wrapper) + ArmSpec registry
│   ├── sensory/                # SensoryPipeline: noise / delay / ablation presets
│   ├── controllers/            # scripted, pd_muscle, motornet, webcam_teleop
│   ├── teleop/                 # webcam hand/body pose tracking + MPL pose mapping
│   ├── training/                # MotorNet-style REINFORCE trainer
│   ├── ui/                     # tkinter control panel
│   └── utils/                  # YAML scenario loader / config dataclasses
└── tests/                      # pytest wrapper around the smoke test
```

Full annotated layout, data-flow diagrams, and "where to edit what" cheat
sheet: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#2-repository-layout).

## Module contracts

* Every controller implements `.reset()` + `.act(obs: Observation) -> np.ndarray[nu]`.
* `SensoryPipeline.observe(raw_state)` returns a named `Observation`
  &mdash; controllers pick fields, never parse indices.
* `MyoArmEnv` follows a gymnasium-like `reset / step` API and exposes
  `state_dim()`, `action_dim()`, `raw_state()`, `state_layout()`.

Because of these contracts you can drop in a new controller (e.g. an
`MPCController`, or a `Bioptim`-based optimal controller) by adding one file in
`src/prosthesis_sim/controllers/` and registering it in the factory &mdash; no
other module needs to change.

## Mapping to the summer-school topic list

* **Forward models / muscle properties / Hill-type**: MyoSuite's
  `myoelbow_2dof6muscles` and `myoarm_bionic_bimanual` models.
* **Proprioception, deafferentation, delays, noise**: `prosthesis_sim.sensory`.
* **Impedance / PD feedback**: `PDMuscleController`.
* **State estimation / prediction (future work)**: extend `sensory` with a
  Kalman filter block.
* **Feedforward / inverse models (future work)**: extend `training/` &mdash; the
  same rollout loop can train a feedforward MLP with a target-driven cost.
* **Prosthesis compensation**: `MotorNetPolicyController` &mdash; a learned RNN
  policy that gets only visual feedback &mdash; and the webcam teleop controller
  as a human-in-the-loop compensation baseline.
* **Real EMG / sensor integration (future work)**: see
  [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#7-sensors-and-real-signal-integration).

## References

* Wang et al., *MyoSim*, ICRA 2022.
* Codol et al., *MotorNet*, eLife 2024 (RP88591).
* Franklin et al., *CNS learns stable, accurate, and efficient movements*, J. Neurosci. 2008.
* Jayasinghe et al., *Somatosensory deafferentation reveals lateralized roles of proprioception*, Curr. Opin. Physiol. 2021.
* Burdet, Franklin, Milner, *Human Robotics: Neuromechanics and Motor Control*, MIT Press 2013.
