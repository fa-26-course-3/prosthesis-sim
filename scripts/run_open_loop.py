"""Reproduce the reference arm_heben demo through the modular stack.

Defaults to the ``human_elbow`` mode (same as the original script). Pass
``--mode prosthesis_elbow`` or ``--mode human_arm`` to launch the same scripted
alternation on a different arm.
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mujoco, mujoco.viewer  # noqa: E402
from prosthesis_sim.controllers import ScriptedController  # noqa: E402
from prosthesis_sim.envs import MyoArmConfig, MyoArmEnv  # noqa: E402
from prosthesis_sim.sensory import SensoryConfig, SensoryPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="human_elbow")
    ap.add_argument("--ball", action="store_true", default=True)
    ap.add_argument("--no-ball", dest="ball", action="store_false")
    args = ap.parse_args()

    env = MyoArmEnv(MyoArmConfig(mode=args.mode, add_ball=args.ball, horizon_s=20.0))
    pipe = SensoryPipeline(SensoryConfig(), dt=env.cfg.timestep, layout=env.state_layout())
    ctrl = ScriptedController.from_env(env, hold_steps=500)

    raw = env.reset()
    print(f"[demo] mode={env.spec.mode}  action_dim={env.action_dim()}  "
          f"flexors={env.flexor_idx.tolist()}  extensors={env.extensor_idx.tolist()}  "
          f"prosth={env.prosth_idx.tolist()}")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            obs = pipe.observe(raw)
            a = ctrl.act(obs)
            raw, _, _, done, _ = env.step(a)
            viewer.sync()
            time.sleep(env.cfg.timestep)
            if done:
                raw = env.reset(); pipe.reset(); ctrl.reset()


if __name__ == "__main__":
    main()
