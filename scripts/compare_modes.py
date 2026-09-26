"""Head-to-head comparison: same target, same controller family, different modes.

Prints one row per (mode, controller, sensory) combination so you can see how
the biological arm and the motorised prosthesis behave under identical
conditions -- the core scientific question of the FA course.
"""
import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from prosthesis_sim.controllers import (  # noqa: E402
    PDMuscleController, ScriptedController, MotorNetPolicyController,
)
from prosthesis_sim.envs import MODES, MyoArmConfig, MyoArmEnv  # noqa: E402
from prosthesis_sim.sensory import SensoryPipeline, preset  # noqa: E402


CONTROLLERS = ["scripted", "pd_muscle", "motornet"]
SENSORY = ["healthy", "deafferented", "visual_only"]


def build_controller(name: str, env):
    if name == "scripted":
        return ScriptedController.from_env(env, hold_steps=200)
    if name == "pd_muscle":
        return PDMuscleController.from_env(env)
    if name == "motornet":
        return MotorNetPolicyController.from_env(env)
    raise ValueError(name)


def run_one(mode: str, ctrl_name: str, sensory_name: str, seed: int = 0) -> dict:
    env = MyoArmEnv(MyoArmConfig(mode=mode, horizon_s=2.5, random_seed=seed))
    pipe = SensoryPipeline(preset(sensory_name), dt=env.cfg.timestep, layout=env.state_layout())
    ctrl = build_controller(ctrl_name, env); ctrl.reset()

    target_rad = np.deg2rad(env.target_deg())
    qposadr = np.array([env.model.jnt_qposadr[j] for j in env.joint_of_interest_idx], dtype=np.int64)

    raw = env.reset(); pipe.reset()
    r_total = 0.0
    while True:
        obs = pipe.observe(raw)
        a = ctrl.act(obs)
        raw, r, _, done, _ = env.step(a)
        r_total += r
        if done:
            break

    qp_final = raw[env.state_layout().qpos][qposadr]
    err_deg = np.rad2deg(qp_final - target_rad)
    return {
        "mode": mode, "controller": ctrl_name, "sensory": sensory_name,
        "reward": r_total, "final_err_deg": err_deg.tolist(),
        "action_dim": env.action_dim(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--controllers", nargs="+", default=CONTROLLERS)
    ap.add_argument("--sensory", nargs="+", default=SENSORY)
    args = ap.parse_args()

    hdr = f"{'mode':<18} {'ctrl':<15} {'sensory':<15} {'nu':>3} {'reward':>9}  final_err_deg"
    print(hdr); print("-" * len(hdr))
    for mode in args.modes:
        for c in args.controllers:
            for s in args.sensory:
                try:
                    r = run_one(mode, c, s)
                    err = [round(x, 2) for x in r["final_err_deg"]]
                    print(f"{r['mode']:<18} {r['controller']:<15} {r['sensory']:<15} "
                          f"{r['action_dim']:>3d} {r['reward']:>9.2f}  {err}")
                except Exception as e:
                    print(f"{mode:<18} {c:<15} {s:<15}  ERROR: {e!r}")
        print()


if __name__ == "__main__":
    main()
