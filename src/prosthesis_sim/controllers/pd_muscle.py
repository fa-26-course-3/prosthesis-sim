"""Joint-space PD controller with muscle / prosthesis projection -- mode-aware.

Works with all shipped modes:

* ``human_elbow``      : 6 muscles split into flexors (BIC*, BRA) and
                         extensors (TRI*). PD torque -> agonist/antagonist
                         activations.
* ``prosthesis_elbow`` : residual muscles clamped by the env; the Exo actuator
                         receives ``|tau| / 3`` in [0, 1].
* ``prosthesis_arm``   : MPL-style *position servos* (see MyoChallenge-2024).
                         PD writes the *target joint angle* directly into the
                         corresponding position actuator (env spec provides the
                         joint -> actuator pairing).  Non-target prosthesis
                         actuators are held at the midpoint of their ctrl range
                         so the hand keeps a neutral pose.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..envs.arm_specs import ArmSpec
from ..sensory import Observation


class PDMuscleController:
    name = "pd_muscle"

    def __init__(self, spec: ArmSpec, action_dim: int,
                 joint_of_interest_idx: np.ndarray, joint_qpos_indices: np.ndarray,
                 flexor_idx: np.ndarray, extensor_idx: np.ndarray,
                 prosth_idx: np.ndarray, target_deg: np.ndarray,
                 pos_servo_pairs: list[tuple[int, int]],
                 ctrl_lo: np.ndarray, ctrl_hi: np.ndarray,
                 kp: float = 8.0, kd: float = 0.6, fallback_activation: float = 0.05):
        self.spec = spec
        self.action_dim = action_dim
        self.joint_of_interest_idx = np.asarray(joint_of_interest_idx, dtype=np.int64)
        self.joint_qpos_indices = np.asarray(joint_qpos_indices, dtype=np.int64)
        self.flexor_idx = np.asarray(flexor_idx, dtype=np.int64)
        self.extensor_idx = np.asarray(extensor_idx, dtype=np.int64)
        self.prosth_idx = np.asarray(prosth_idx, dtype=np.int64)
        self.pos_servo_pairs = list(pos_servo_pairs)
        self.ctrl_lo = np.asarray(ctrl_lo, dtype=np.float32)
        self.ctrl_hi = np.asarray(ctrl_hi, dtype=np.float32)
        self.target = np.deg2rad(np.asarray(target_deg, dtype=np.float32))
        self.kp = float(kp)
        self.kd = float(kd)
        self.fallback = float(fallback_activation)

    @classmethod
    def from_env(cls, env, **kw) -> "PDMuscleController":
        qposadr = np.array(
            [env.model.jnt_qposadr[j] for j in env.joint_of_interest_idx],
            dtype=np.int64,
        )
        return cls(env.spec, env.action_dim(), env.joint_of_interest_idx,
                   qposadr, env.flexor_idx, env.extensor_idx, env.prosth_idx,
                   target_deg=env.target_deg(),
                   pos_servo_pairs=env.pos_servo_pairs,
                   ctrl_lo=env._ctrl_lo, ctrl_hi=env._ctrl_hi, **kw)

    def reset(self) -> None:
        pass

    def _neutral_action(self) -> np.ndarray:
        """Midpoint of each actuator's ctrl range (position servos), fallback
        low tone otherwise."""
        u = 0.5 * (self.ctrl_lo + self.ctrl_hi)
        # For actuators whose range crosses zero (typical muscle [0.001, 1]),
        # prefer the fallback activation instead of the midpoint 0.5.
        muscle_mask = (self.ctrl_lo >= 0.0) & (self.ctrl_hi <= 1.0)
        u[muscle_mask] = self.fallback
        return u.astype(np.float32)

    def act(self, obs: Observation, target: Optional[np.ndarray] = None) -> np.ndarray:
        tgt = self.target if target is None else np.deg2rad(np.asarray(target, dtype=np.float32))
        u = self._neutral_action()

        # --- Position-servo path (e.g. MPL prosthesis_arm) ------------------
        # Write target angle straight into the servo. The servo does the PD
        # internally; we only need to keep the command inside the ctrl range.
        if self.pos_servo_pairs:
            for k, (_joint_idx, act_idx) in enumerate(self.pos_servo_pairs):
                if k >= len(tgt):
                    break
                lo, hi = self.ctrl_lo[act_idx], self.ctrl_hi[act_idx]
                u[act_idx] = float(np.clip(tgt[k], lo, hi))
            return u

        # --- Torque-projection paths (elbow modes) --------------------------
        if not obs.proprio_available:
            if len(self.prosth_idx) > 0:
                u[self.prosth_idx] = self.fallback
            return u

        try:
            q = obs.qpos[self.joint_qpos_indices]
            qd = obs.qvel[self.joint_qpos_indices]
        except IndexError:
            q = obs.qpos[: len(tgt)]
            qd = obs.qvel[: len(tgt)]

        err = tgt - q
        tau = self.kp * err - self.kd * qd

        # Prosthesis_elbow (single Exo motor).
        if len(self.prosth_idx) > 0 and len(tau) > 0:
            u[self.prosth_idx] = float(np.clip(tau[0] / 3.0, 0.0, 1.0))
            return u

        # Muscle mode (human_elbow): agonist/antagonist projection on primary joint.
        tau_primary = float(tau[-1])
        if tau_primary > 0:
            u[self.flexor_idx] = np.clip(tau_primary / 5.0, 0.0, 1.0)
        else:
            u[self.extensor_idx] = np.clip(-tau_primary / 5.0, 0.0, 1.0)
        return u.astype(np.float32)
