"""Minimal MotorNet-style training loop -- mode-aware.

Trains :class:`SmallRNNPolicy` sized to the current mode's action_dim, then saves
weights under ``models/motornet_policy_<mode>.pt``.  Uses REINFORCE because
MuJoCo's ``mj_step`` is not autograd-differentiable; swap ``MyoArmEnv`` for a
``motornet.effector`` to obtain a fully differentiable rollout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from prosthesis_sim.controllers.motornet_policy import SmallRNNPolicy, MODELS_DIR
from prosthesis_sim.envs import DEFAULT_MODE, MyoArmConfig, MyoArmEnv
from prosthesis_sim.sensory import SensoryConfig, SensoryPipeline


def rollout(env: MyoArmEnv, pipe: SensoryPipeline, policy: SmallRNNPolicy,
            target: np.ndarray, T: int, device: torch.device):
    raw = env.reset(); pipe.reset()
    h = None
    log_probs = []
    rewards = []
    for _ in range(T):
        obs = pipe.observe(raw)
        vec = np.concatenate([obs.hand_pos, target]).astype(np.float32)
        x = torch.from_numpy(vec).view(1, 1, -1).to(device)
        mean, h = policy(x, h)
        dist = torch.distributions.Normal(mean.view(-1), 0.1)
        action = dist.sample()
        log_probs.append(dist.log_prob(action).sum())
        a = action.detach().cpu().numpy().clip(0.0, 1.0)
        raw, r, _, done, _ = env.step(a)
        rewards.append(r)
        if done:
            break
    return log_probs, rewards


def train(
    mode: str = DEFAULT_MODE,
    episodes: int = 60,
    horizon_s: float = 2.0,
    lr: float = 3e-3,
    seed: int = 0,
    out_path: str | Path | None = None,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = MyoArmEnv(MyoArmConfig(mode=mode, horizon_s=horizon_s, random_seed=seed))
    pipe = SensoryPipeline(
        SensoryConfig(proprio_enabled=False, vision_delay_ms=120.0),
        dt=env.cfg.timestep,
        layout=env.state_layout(),
    )
    device = torch.device("cpu")
    policy = SmallRNNPolicy(action_dim=env.action_dim()).to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)

    raw0 = env.reset()
    hand0 = raw0[env.state_layout().hand]
    target = hand0 + np.array([0.02, -0.05, -0.03], dtype=np.float32)

    T = int(horizon_s / env.cfg.timestep)
    baseline = 0.0
    ema = 0.9
    for ep in range(episodes):
        log_probs, rewards = rollout(env, pipe, policy, target, T, device)
        R = float(sum(rewards))
        baseline = ema * baseline + (1 - ema) * R
        adv = R - baseline
        loss = -torch.stack(log_probs).sum() * adv
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if (ep + 1) % 10 == 0:
            print(f"[train:{mode}] ep {ep+1:3d}  return={R:8.2f}  baseline={baseline:8.2f}")

    out = Path(out_path) if out_path else (MODELS_DIR / f"motornet_policy_{mode}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out)
    print(f"[train:{mode}] saved -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=DEFAULT_MODE)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--horizon-s", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    train(args.mode, args.episodes, args.horizon_s, args.lr, args.seed, args.out)


if __name__ == "__main__":
    main()
