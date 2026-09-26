"""MotorNet-style GRU policy -- action dimension resolved from the env.

Takes ``(current_hand xyz, target xyz)`` as the recurrent input and outputs a
per-actuator activation vector of length ``env.action_dim()`` (6, 7, or 14
depending on the mode).  A per-mode weights file lives at
``models/motornet_policy_<mode>.pt``; if missing, the policy still runs but
starts from a low-activation prior (biased sigmoid).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..envs.arm_specs import ArmSpec
from ..sensory import Observation


MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


class SmallRNNPolicy(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, action_dim: int = 6):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, action_dim)
        nn.init.constant_(self.head.bias, -1.0)
        self.action_dim = action_dim

    def forward(self, x, h=None):
        y, h = self.gru(x, h)
        return torch.sigmoid(self.head(y)), h


class MotorNetPolicyController:
    name = "motornet_policy"

    def __init__(self, spec: ArmSpec, action_dim: int,
                 ctrl_lo: np.ndarray, ctrl_hi: np.ndarray,
                 weights_path: Optional[str] = None,
                 target_hand_xyz: tuple = (0.35, -0.25, 1.20),
                 device: str = "cpu"):
        self.spec = spec
        self.action_dim = int(action_dim)
        self.ctrl_lo = np.asarray(ctrl_lo, dtype=np.float32)
        self.ctrl_hi = np.asarray(ctrl_hi, dtype=np.float32)
        self.device = torch.device(device)
        self.policy = SmallRNNPolicy(action_dim=self.action_dim).to(self.device).eval()
        self.weights_path = Path(weights_path) if weights_path else (MODELS_DIR / f"motornet_policy_{spec.mode}.pt")
        self._loaded = False
        if self.weights_path.exists():
            try:
                sd = torch.load(self.weights_path, map_location=self.device)
                self.policy.load_state_dict(sd)
                self._loaded = True
            except Exception:
                self._loaded = False
        self.target = np.asarray(target_hand_xyz, dtype=np.float32)
        self._h = None

    @classmethod
    def from_env(cls, env, **kw) -> "MotorNetPolicyController":
        return cls(env.spec, env.action_dim(),
                   ctrl_lo=env._ctrl_lo, ctrl_hi=env._ctrl_hi, **kw)

    def reset(self) -> None:
        self._h = None

    def act(self, obs: Observation, target: Optional[np.ndarray] = None) -> np.ndarray:
        tgt = self.target if target is None else np.asarray(target, dtype=np.float32)
        vec = np.concatenate([obs.hand_pos, tgt]).astype(np.float32)
        x = torch.from_numpy(vec).view(1, 1, -1).to(self.device)
        with torch.no_grad():
            y, self._h = self.policy(x, self._h)
        u01 = y.view(-1).cpu().numpy().astype(np.float32)      # sigmoid in [0, 1]
        return (self.ctrl_lo + u01 * (self.ctrl_hi - self.ctrl_lo)).astype(np.float32)

    @property
    def loaded_pretrained(self) -> bool:
        return self._loaded
