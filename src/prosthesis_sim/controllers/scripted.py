"""Open-loop scripted controller -- mode-aware.

Alternates between two fixed action patterns according to the :class:`ArmSpec`.
This is the modular equivalent of ``_documents/MyoSim_MyoSuite_MoJoCo/arm_heben.py``
that works unchanged across all modes:

* ``human_elbow``      : classic agonist/antagonist alternation (3+3 muscles).
* ``prosthesis_elbow`` : Exo motor alternates between low and high activation.
* ``prosthesis_arm``   : primary position-servo alternates between the low and
                         high ends of its ctrl range (equivalent to
                         ``elbow=0.05 rad`` vs ``elbow=2.35 rad``); the rest of
                         the hand stays at the ctrl-range midpoint.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..envs.arm_specs import ArmSpec
from ..sensory import Observation


class ScriptedController:
    name = "scripted"

    def __init__(self, spec: ArmSpec, action_dim: int,
                 flexor_idx: np.ndarray, extensor_idx: np.ndarray,
                 prosth_idx: np.ndarray, pos_servo_pairs: list[tuple[int, int]],
                 ctrl_lo: np.ndarray, ctrl_hi: np.ndarray,
                 hold_steps: int = 500, high: float = 1.0, low: float = 0.0):
        self.spec = spec
        self.action_dim = action_dim
        self.flexor_idx = np.asarray(flexor_idx, dtype=np.int64)
        self.extensor_idx = np.asarray(extensor_idx, dtype=np.int64)
        self.prosth_idx = np.asarray(prosth_idx, dtype=np.int64)
        self.pos_servo_pairs = list(pos_servo_pairs)
        self.ctrl_lo = np.asarray(ctrl_lo, dtype=np.float32)
        self.ctrl_hi = np.asarray(ctrl_hi, dtype=np.float32)
        self.hold_steps = hold_steps
        self.high, self.low = float(high), float(low)
        self._i = 0
        self._neutral = 0.5 * (self.ctrl_lo + self.ctrl_hi)
        muscle_mask = (self.ctrl_lo >= 0.0) & (self.ctrl_hi <= 1.0)
        self._neutral[muscle_mask] = 0.05     # resting muscle tone

    @classmethod
    def from_env(cls, env, **kw) -> "ScriptedController":
        return cls(env.spec, env.action_dim(), env.flexor_idx, env.extensor_idx,
                   env.prosth_idx, env.pos_servo_pairs,
                   env._ctrl_lo, env._ctrl_hi, **kw)

    def reset(self) -> None:
        self._i = 0

    def act(self, obs: Observation, target: Optional[np.ndarray] = None) -> np.ndarray:
        flex_phase = (self._i // self.hold_steps) % 2 == 0
        u = self._neutral.copy()

        # Position-servo path: swing primary joint between low and high extremes.
        if self.pos_servo_pairs:
            for _joint_idx, act_idx in self.pos_servo_pairs:
                lo, hi = self.ctrl_lo[act_idx], self.ctrl_hi[act_idx]
                u[act_idx] = hi if flex_phase else lo
            self._i += 1
            return u

        # Activation-space (muscles / motor / Exo).
        if len(self.prosth_idx) > 0:
            u[self.prosth_idx] = self.high if flex_phase else self.low
        if flex_phase:
            u[self.flexor_idx] = self.high
            u[self.extensor_idx] = self.low
        else:
            u[self.flexor_idx] = self.low
            u[self.extensor_idx] = self.high
        self._i += 1
        return u.astype(np.float32)
