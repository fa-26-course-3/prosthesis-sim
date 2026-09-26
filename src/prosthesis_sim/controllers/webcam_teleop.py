"""Controller that mirrors a real webcam-tracked hand onto the MPL prosthesis.

Contract identical to any other controller: implements
``.reset()`` and ``.act(obs, target=None) -> np.ndarray[nu]`` so it can be
dropped into every existing loop (scripts, GUI, smoke test).

Internally it owns a background :class:`HandTracker`; ``.act()`` never blocks
on webcam I/O.  When the tracker can't see a hand the last valid target is
held (safe hover), never NaN.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..envs.arm_specs import ArmSpec
from ..sensory import Observation
from ..teleop import HandTracker, PoseMapper, make_tracker


class WebcamTeleopController:
    name = "webcam_teleop"

    def __init__(self, spec: ArmSpec, action_dim: int,
                 act_idx: dict, pos_servo_pairs: list[tuple[int, int]],
                 ctrl_lo: np.ndarray, ctrl_hi: np.ndarray,
                 tracker: Optional[HandTracker] = None,
                 tracker_kind: str = "mediapipe",
                 tracker_kwargs: Optional[dict] = None,
                 mapper: Optional[PoseMapper] = None):
        self.spec = spec
        self.action_dim = int(action_dim)
        self.act_idx = dict(act_idx)                     # name -> actuator index
        self.pos_servo_pairs = list(pos_servo_pairs)     # (joint_idx, act_idx)
        self.ctrl_lo = np.asarray(ctrl_lo, dtype=np.float32)
        self.ctrl_hi = np.asarray(ctrl_hi, dtype=np.float32)
        self.tracker = tracker if tracker is not None else make_tracker(
            tracker_kind, **(tracker_kwargs or {})
        )
        self.mapper = mapper if mapper is not None else PoseMapper()
        self._started = False
        self._neutral = self._compute_neutral()

    @classmethod
    def from_env(cls, env, **kw) -> "WebcamTeleopController":
        return cls(env.spec, env.action_dim(), env._act_idx,
                   env.pos_servo_pairs, env._ctrl_lo, env._ctrl_hi, **kw)

    # ----------------------------------------------------------------- api
    def reset(self) -> None:
        if not self._started:
            self.tracker.start()
            self._started = True
        self.mapper.reset()

    def act(self, obs: Observation, target: Optional[np.ndarray] = None) -> np.ndarray:
        pose = self.tracker.latest()
        body = self.tracker.latest_body()
        targets = self.mapper.map(pose, body)     # {joint_name: rad}
        u = self._neutral.copy()
        for name, val in targets.items():
            idx = self.act_idx.get(name)
            if idx is None:
                continue
            u[idx] = float(np.clip(val, self.ctrl_lo[idx], self.ctrl_hi[idx]))
        return u.astype(np.float32)

    def close(self) -> None:
        if self._started:
            try:
                self.tracker.stop()
            except Exception:
                pass
        self._started = False

    # ------------------------------------------------------------ helpers
    def _compute_neutral(self) -> np.ndarray:
        u = 0.5 * (self.ctrl_lo + self.ctrl_hi)
        muscle_mask = (self.ctrl_lo >= 0.0) & (self.ctrl_hi <= 1.0)
        u[muscle_mask] = 0.05
        return u.astype(np.float32)
