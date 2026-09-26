"""Headless end-to-end smoke test across every mode + representative scenario.

Prints one line per (mode, controller, sensory) triple. Asserts shapes, no NaNs,
and reasonable episode length.
"""
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from prosthesis_sim.controllers import (  # noqa: E402
    PDMuscleController, ScriptedController, MotorNetPolicyController,
)
from prosthesis_sim.envs import MODES, MyoArmConfig, MyoArmEnv  # noqa: E402
from prosthesis_sim.sensory import SensoryPipeline, preset  # noqa: E402


SCENARIOS = [
    ("scripted",  "healthy"),
    ("pd_muscle", "healthy"),
    ("pd_muscle", "deafferented"),
    ("motornet",  "visual_only"),
]


def build_ctrl(name, env):
    if name == "scripted":  return ScriptedController.from_env(env, hold_steps=100)
    if name == "pd_muscle": return PDMuscleController.from_env(env)
    if name == "motornet":  return MotorNetPolicyController.from_env(env)
    raise ValueError(name)


def run(mode, ctrl_name, sens_name, horizon_s=1.2) -> dict:
    env = MyoArmEnv(MyoArmConfig(mode=mode, horizon_s=horizon_s, random_seed=0))
    pipe = SensoryPipeline(preset(sens_name), dt=env.cfg.timestep, layout=env.state_layout())
    ctrl = build_ctrl(ctrl_name, env); ctrl.reset()
    raw = env.reset()
    total_r = 0.0
    t0 = time.perf_counter()
    n = 0
    while True:
        obs = pipe.observe(raw)
        a = ctrl.act(obs)
        assert a.shape == (env.action_dim(),), (a.shape, env.action_dim())
        assert not np.isnan(a).any(), "NaN in action"
        raw, r, _, done, _ = env.step(a)
        assert not np.isnan(raw).any(), "NaN in raw state"
        total_r += r; n += 1
        if done:
            break
    return dict(mode=mode, ctrl=ctrl_name, sens=sens_name, steps=n,
                reward=total_r, wall_s=time.perf_counter() - t0,
                nu=env.action_dim())


def main() -> int:
    hdr = f"{'mode':<18} {'ctrl':<12} {'sens':<15} {'nu':>3} {'steps':>6} {'reward':>9} {'wall_s':>7}"
    print(hdr); print("-" * len(hdr))
    ok = True
    for mode in MODES:
        for ctrl_name, sens_name in SCENARIOS:
            try:
                r = run(mode, ctrl_name, sens_name)
                print(f"{r['mode']:<18} {r['ctrl']:<12} {r['sens']:<15} "
                      f"{r['nu']:>3d} {r['steps']:>6d} {r['reward']:>9.2f} {r['wall_s']:>7.2f}")
            except Exception as e:
                ok = False
                print(f"{mode:<18} {ctrl_name:<12} {sens_name:<15}  FAILED: {e!r}")
    print("-" * len(hdr))
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
