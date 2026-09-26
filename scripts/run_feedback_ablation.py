"""Sweep sensory scenarios headless and print reach-error metrics.

Extended for the mode system: pass ``--mode`` (or ``--all-modes``) to compare
human elbow, full human arm, and prosthesis modes on the same PD baseline.
"""
import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from prosthesis_sim.controllers import PDMuscleController  # noqa: E402
from prosthesis_sim.envs import MODES, MyoArmConfig, MyoArmEnv  # noqa: E402
from prosthesis_sim.sensory import SensoryPipeline, preset  # noqa: E402


PRESETS = ["healthy", "deafferented", "noisy_proprio", "delayed_proprio", "visual_only"]


def run_one(mode: str, sensory_name: str, seed: int = 0) -> dict:
    env = MyoArmEnv(MyoArmConfig(mode=mode, horizon_s=3.0, random_seed=seed))
    pipe = SensoryPipeline(preset(sensory_name), dt=env.cfg.timestep, layout=env.state_layout())
    ctrl = PDMuscleController.from_env(env)

    target_rad = np.deg2rad(env.target_deg())
    qposadr = np.array([env.model.jnt_qposadr[j] for j in env.joint_of_interest_idx], dtype=np.int64)

    raw = env.reset(); pipe.reset(); ctrl.reset()
    err_hist = []
    r_total = 0.0
    while True:
        obs = pipe.observe(raw)
        a = ctrl.act(obs)
        raw, r, _, done, _ = env.step(a)
        r_total += r
        qp = raw[env.state_layout().qpos][qposadr]
        err_hist.append(np.linalg.norm(target_rad - qp))
        if done:
            break
    err = np.asarray(err_hist)
    return {
        "mode": mode,
        "scenario": sensory_name,
        "reward": r_total,
        "final_err_rad": float(err[-1]),
        "mean_err_rad": float(err.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None,
                    help="Single arm mode (see prosthesis_sim.envs.MODES).")
    ap.add_argument("--all-modes", action="store_true",
                    help="Iterate over every registered mode.")
    args = ap.parse_args()

    modes = list(MODES) if args.all_modes or args.mode is None else [args.mode]
    header = f"{'mode':<18} {'scenario':<20} {'reward':>10} {'final_err':>10} {'mean_err':>10}"
    print(header)
    print("-" * len(header))
    for mode in modes:
        for name in PRESETS:
            r = run_one(mode, name)
            print(f"{r['mode']:<18} {r['scenario']:<20} {r['reward']:>10.2f} "
                  f"{r['final_err_rad']:>10.4f} {r['mean_err_rad']:>10.4f}")
        print()


if __name__ == "__main__":
    main()
