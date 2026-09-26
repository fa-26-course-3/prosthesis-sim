"""Configurable sensory feedback pipeline.

Sits between :class:`prosthesis_sim.envs.MyoArmEnv` and any controller. It reads
its slice map from the env (see :class:`prosthesis_sim.envs.StateLayout`) so the
same pipeline works with the 6-muscle elbow model, the 14-actuator full arm,
and the 7-actuator prosthesis mode -- all without code changes.

Ablation knobs (:class:`SensoryConfig`):
* proprioception (spindle + GTO) : on/off, noise, delay
* vision (endpoint xyz)          : on/off, noise, delay
* tactile (contact placeholder)  : on/off
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np


@dataclass
class SensoryConfig:
    proprio_enabled: bool = True
    proprio_noise_std: float = 0.0
    proprio_delay_ms: float = 0.0
    vision_enabled: bool = True
    vision_delay_ms: float = 100.0
    vision_noise_std: float = 0.0
    tactile_enabled: bool = True
    seed: Optional[int] = None


@dataclass
class Observation:
    qpos: np.ndarray            # (nq,)
    qvel: np.ndarray            # (nv,)
    muscle_len: np.ndarray      # (nu,)  actuator lengths
    muscle_vel: np.ndarray      # (nu,)  actuator velocities
    hand_pos: np.ndarray        # (3,)
    proprio_available: bool = True
    vision_available: bool = True
    t: float = 0.0

    def as_vector(self) -> np.ndarray:
        return np.concatenate([
            self.qpos, self.qvel, self.muscle_len, self.muscle_vel, self.hand_pos,
        ]).astype(np.float32)


class SensoryPipeline:
    """Turns raw physics state into a degraded :class:`Observation`.

    The state layout is not fixed at import time -- it comes from
    :meth:`MyoArmEnv.state_layout`, so the same pipeline handles any arm mode.
    """

    def __init__(self, cfg: SensoryConfig, dt: float, layout=None):
        self.cfg = cfg
        self.dt = dt
        self.layout = layout
        self._rng = np.random.default_rng(cfg.seed)
        self._proprio_buf: Deque[np.ndarray] = deque()
        self._vision_buf: Deque[np.ndarray] = deque()
        self._t = 0.0

    # ------------------------------------------------------------------
    def bind(self, layout) -> None:
        """Attach an env's StateLayout after construction (used by the GUI)."""
        self.layout = layout
        self.reset()

    def reset(self) -> None:
        self._proprio_buf.clear()
        self._vision_buf.clear()
        self._t = 0.0

    def _delay_steps(self, delay_ms: float) -> int:
        return max(0, int(round((delay_ms * 1e-3) / self.dt)))

    def _push_delayed(self, buf: Deque[np.ndarray], value: np.ndarray, delay_ms: float) -> np.ndarray:
        n = self._delay_steps(delay_ms)
        buf.append(value.copy())
        while len(buf) > n + 1:
            buf.popleft()
        return buf[0]

    # ------------------------------------------------------------------
    def observe(self, raw_state: np.ndarray) -> Observation:
        assert self.layout is not None, "SensoryPipeline needs a StateLayout (pass at construction or via .bind())."
        L = self.layout
        cfg = self.cfg

        qpos = raw_state[L.qpos]
        qvel = raw_state[L.qvel]
        mlen = raw_state[L.act_len]
        mvel = raw_state[L.act_vel]
        hand = raw_state[L.hand]

        # --- Proprioception --------------------------------------------------
        if cfg.proprio_enabled:
            prop_raw = np.concatenate([qpos, qvel, mlen, mvel])
            if cfg.proprio_noise_std > 0.0:
                prop_raw = prop_raw + self._rng.normal(0.0, cfg.proprio_noise_std, prop_raw.shape)
            prop_delayed = self._push_delayed(self._proprio_buf, prop_raw, cfg.proprio_delay_ms)
            qpos_o = prop_delayed[: L.nq]
            qvel_o = prop_delayed[L.nq : L.nq + L.nv]
            mlen_o = prop_delayed[L.nq + L.nv : L.nq + L.nv + L.nu]
            mvel_o = prop_delayed[L.nq + L.nv + L.nu :]
        else:
            qpos_o = np.zeros(L.nq, dtype=np.float32)
            qvel_o = np.zeros(L.nv, dtype=np.float32)
            mlen_o = np.zeros(L.nu, dtype=np.float32)
            mvel_o = np.zeros(L.nu, dtype=np.float32)

        # --- Vision ---------------------------------------------------------
        if cfg.vision_enabled:
            v = hand.copy()
            if cfg.vision_noise_std > 0.0:
                v = v + self._rng.normal(0.0, cfg.vision_noise_std, v.shape)
            hand_o = self._push_delayed(self._vision_buf, v, cfg.vision_delay_ms)
        else:
            hand_o = np.zeros(3, dtype=np.float32)

        obs = Observation(
            qpos=np.asarray(qpos_o, dtype=np.float32),
            qvel=np.asarray(qvel_o, dtype=np.float32),
            muscle_len=np.asarray(mlen_o, dtype=np.float32),
            muscle_vel=np.asarray(mvel_o, dtype=np.float32),
            hand_pos=np.asarray(hand_o, dtype=np.float32),
            proprio_available=cfg.proprio_enabled,
            vision_available=cfg.vision_enabled,
            t=self._t,
        )
        self._t += self.dt
        return obs


# -------------------------------------------------------------- named presets --
def preset(name: str) -> SensoryConfig:
    name = name.lower().strip()
    if name in ("healthy", "baseline", "full"):
        return SensoryConfig()
    if name in ("deafferented", "no_proprio", "no_feedback"):
        return SensoryConfig(proprio_enabled=False)
    if name in ("noisy_proprio", "noisy"):
        return SensoryConfig(proprio_noise_std=0.02, proprio_delay_ms=20.0)
    if name in ("delayed_proprio", "delayed"):
        return SensoryConfig(proprio_delay_ms=50.0)
    if name in ("visual_only", "prosthesis"):
        return SensoryConfig(proprio_enabled=False, vision_delay_ms=120.0)
    raise ValueError(f"unknown sensory preset '{name}'")
