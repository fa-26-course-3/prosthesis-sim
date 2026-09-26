"""MuJoCo/MyoSuite environment wrapper -- **mode-aware**.

The env is parameterised by an :class:`ArmSpec` (see :mod:`arm_specs`) so the
same class runs the biological ``human_elbow`` model, the full ``human_arm``
model, or the ``prosthesis_elbow`` model with a motorised Exo joint.

All the rest of the stack (sensory pipeline, controllers, GUI) reads its
geometry information (joint indices, muscle / prosthesis actuator indices,
target angles) from ``env.spec`` and ``env.state_layout()`` -- no more hard-coded
constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco
import myosuite  # noqa: F401
import numpy as np

from .arm_specs import ArmSpec, DEFAULT_MODE, get_spec, TARGET_POSES


@dataclass
class StateLayout:
    """Byte-slice map of an env's ``raw_state()`` for use by SensoryPipeline."""
    qpos: slice
    qvel: slice
    act_len: slice        # actuator_length (muscle length for Hill actuators)
    act_vel: slice
    hand: slice
    nq: int
    nv: int
    nu: int


@dataclass
class MyoArmConfig:
    """Environment-level configuration."""

    mode: str = DEFAULT_MODE
    xml_path_override: Optional[str] = None
    timestep: float = 0.002
    horizon_s: float = 6.0
    add_ball: bool = False
    ball_mass_kg: float = 2.0
    ball_radius_m: float = 0.06
    target_deg_override: Optional[Tuple[float, ...]] = None
    target_pose: Optional[str] = None             # name from arm_specs.TARGET_POSES
    prosthesis_muscle_tone: float = 0.05
    # Button-press manipulation task (MyoChallenge-style with a button instead
    # of the jelly box). Only used when add_button=True.
    add_button: bool = False
    # Default position matches the palm endpoint in the ``reach_forward`` pose
    # so the button fires the moment the servo settles into that target angle.
    button_pos_world: Tuple[float, float, float] = (0.315, -0.224, 1.35)
    button_radius_m: float = 0.05
    button_travel_m: float = 0.02
    button_contact_threshold_N: float = 2.0
    random_seed: Optional[int] = None


class MyoArmEnv:
    """Mode-aware wrapper around MyoSuite arm/elbow/prosthesis models."""

    def __init__(self, cfg: Optional[MyoArmConfig] = None):
        self.cfg = cfg or MyoArmConfig()
        self.spec: ArmSpec = get_spec(self.cfg.mode)
        # Resolve a target-pose name into explicit joint-target degrees.
        if self.cfg.target_pose is not None and self.cfg.target_deg_override is None:
            if self.cfg.target_pose not in TARGET_POSES:
                raise KeyError(
                    f"unknown target_pose '{self.cfg.target_pose}'. Available: {sorted(TARGET_POSES)}"
                )
            self.cfg = replace_config_targets(self.cfg, TARGET_POSES[self.cfg.target_pose])
        self._rng = np.random.default_rng(self.cfg.random_seed)
        self._build_model()
        self._resolve_indices()
        self._layout = self._compute_layout()
        self.button_pressed = False
        self.button_press_step = -1
        self.reset()

    # ---------------------------------------------------------------- build
    def _build_model(self) -> None:
        xml = self.cfg.xml_path_override or self.spec.resolve_xml()
        spec = mujoco.MjSpec.from_file(xml)
        if self.cfg.add_ball and self.spec.ball_parent_body is not None:
            try:
                parent = spec.body(self.spec.ball_parent_body)
                ball = parent.add_body(name="held_ball", pos=[0.0, -0.30, 0.0])
                ball.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[self.cfg.ball_radius_m, 0.0, 0.0],
                    mass=self.cfg.ball_mass_kg,
                    rgba=[1.0, 0.5, 0.0, 1.0],
                )
            except Exception as e:
                print(f"[MyoArmEnv] could not attach ball to '{self.spec.ball_parent_body}': {e!r}")
        if self.cfg.add_button:
            self._attach_button(spec)
        # Grow the MuJoCo model arena so heavy scenes (bimanual + injected
        # geoms) compile without "Could not allocate memory". Passing bytes
        # (100 MB) is well above the default and cheap on modern machines.
        try:
            spec.memory = max(int(getattr(spec, "memory", 0) or 0), 100 * 1024 * 1024)
        except Exception:
            pass
        self.model = spec.compile()
        self.model.opt.timestep = self.cfg.timestep
        self.data = mujoco.MjData(self.model)
        self.n_steps = int(self.cfg.horizon_s / self.cfg.timestep)
        self._step_count = 0

    def _attach_button(self, spec) -> None:
        """Add a red button disc floating in world space at the configured pos.

        The button is a single static red cylinder (no pedestal) placed exactly
        where the MPL palm should land, so a contact fires as the palm swings
        in.  Contact detection is done in :meth:`_check_button_press`.
        """
        try:
            world = spec.worldbody
            base = world.add_body(name="button_base", pos=list(self.cfg.button_pos_world))
            base.add_geom(
                name="button_top",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[self.cfg.button_radius_m, 0.012, 0.0],
                rgba=[0.9, 0.1, 0.1, 1.0],
                # Permissive contact masks (MPL hand geoms use contype=2).
                contype=3, conaffinity=3,
            )
        except Exception as e:
            print(f"[MyoArmEnv] could not attach button: {e!r}")

    # ---------------------------------------------------------- index cache
    def _resolve_indices(self) -> None:
        m = self.model
        self._act_idx: dict[str, int] = {}
        for i in range(m.nu):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name is not None:
                self._act_idx[name] = i

        def _lookup(names):
            return np.array([self._act_idx[n] for n in names if n in self._act_idx], dtype=np.int64)

        self.flexor_idx    = _lookup(self.spec.flexor_actuators)
        self.extensor_idx  = _lookup(self.spec.extensor_actuators)
        self.prosth_idx    = _lookup(self.spec.prosthetic_actuators)
        self.bio_idx       = _lookup(self.spec.biological_actuators)

        self._joint_idx: dict[str, int] = {}
        for i in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name is not None:
                self._joint_idx[name] = i
        self.joint_of_interest_idx = np.array(
            [self._joint_idx[j] for j in self.spec.joints_of_interest if j in self._joint_idx],
            dtype=np.int64,
        )

        # Position-servo mapping (joint -> actuator index) for MPL-style arms.
        self.pos_servo_pairs: list[tuple[int, int]] = []
        for joint_name, act_name in self.spec.prosth_position_actuators:
            if joint_name in self._joint_idx and act_name in self._act_idx:
                self.pos_servo_pairs.append((self._joint_idx[joint_name], self._act_idx[act_name]))

        # Per-actuator ctrl bounds (respected in step() when actuator_ctrllimited is set).
        self._ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
        # For unlimited actuators default to a safe [0, 1] window (matches all
        # Hill muscles); the Exo motor in prosthesis_elbow lands here.
        unlimited = (self.model.actuator_ctrllimited == 0)
        self._ctrl_lo[unlimited] = 0.0
        self._ctrl_hi[unlimited] = 1.0

        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.spec.endpoint_body)
        self._endpoint_bid = bid if bid >= 0 else m.nbody - 1

        # Button-press manipulation task bookkeeping.
        self._button_top_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "button_top")
        self._button_base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "button_base")
        # Endpoint contact-geom set: palm + all finger phalanges so a fingertip
        # tap counts as a valid press, not only a palm slap.
        self._endpoint_geoms = self._prosthesis_hand_geoms()

    def _compute_layout(self) -> StateLayout:
        m = self.model
        nq, nv, nu = m.nq, m.nv, m.nu
        return StateLayout(
            qpos=slice(0, nq),
            qvel=slice(nq, nq + nv),
            act_len=slice(nq + nv, nq + nv + nu),
            act_vel=slice(nq + nv + nu, nq + nv + 2 * nu),
            hand=slice(nq + nv + 2 * nu, nq + nv + 2 * nu + 3),
            nq=nq, nv=nv, nu=nu,
        )

    # ------------------------------------------------------------------ api
    def state_layout(self) -> StateLayout:
        return self._layout

    def state_dim(self) -> int:
        L = self._layout
        return L.nq + L.nv + 2 * L.nu + 3

    def action_dim(self) -> int:
        return self.model.nu

    def target_deg(self) -> np.ndarray:
        if self.cfg.target_deg_override is not None:
            return np.asarray(self.cfg.target_deg_override, dtype=np.float32)
        return np.asarray(self.spec.joint_targets_deg, dtype=np.float32)

    # ---------------------------------------------------------------- reset
    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        if self.cfg.random_seed is not None:
            self.data.qpos[:] += self._rng.normal(scale=0.005, size=self.data.qpos.shape)
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self.button_pressed = False
        self.button_press_step = -1
        if self._button_top_gid >= 0:
            self.model.geom_rgba[self._button_top_gid] = [0.9, 0.1, 0.1, 1.0]
        return self.raw_state()

    # ---------------------------------------------------------------- step
    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float64)
        # Clip each actuator to its declared MuJoCo ctrl range (falls back to
        # [0, 1] for unlimited actuators -- see _resolve_indices).
        a = np.clip(a, self._ctrl_lo, self._ctrl_hi)
        # Residual muscles under prosthesis: overridden to a constant low tone.
        if self.spec.is_prosthesis and len(self.bio_idx) > 0:
            a[self.bio_idx] = self.cfg.prosthesis_muscle_tone
        self.data.ctrl[: self.model.nu] = a
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        just_pressed = self._check_button_press()
        state = self.raw_state()
        reward = self._reward(state, a) + (5.0 if just_pressed else 0.0)
        truncated = self._step_count >= self.n_steps
        info = {
            "target_deg": self.target_deg(),
            "joint_of_interest_idx": self.joint_of_interest_idx.tolist(),
            "mode": self.spec.mode,
            "button_pressed": self.button_pressed,
            "button_press_step": self.button_press_step,
            "button_pos_world": (
                self.data.xpos[self._button_base_bid].tolist()
                if self._button_base_bid >= 0 else None
            ),
            "endpoint_pos_world": self._hand_pos().tolist(),
        }
        return state, reward, False, truncated, info

    def _check_button_press(self) -> bool:
        """Detect palm/finger contact with the red button top.

        Returns True on the *first* frame a contact above the force threshold
        is seen (used as a one-shot reward bonus).
        """
        if self.button_pressed or self._button_top_gid < 0:
            return False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self._button_top_gid or c.geom2 == self._button_top_gid:
                other = c.geom2 if c.geom1 == self._button_top_gid else c.geom1
                if other in self._endpoint_geoms:
                    self.button_pressed = True
                    self.button_press_step = self._step_count
                    self.model.geom_rgba[self._button_top_gid] = [0.1, 0.9, 0.1, 1.0]
                    return True
        return False

    def _geoms_of_body(self, bid: int) -> set[int]:
        """All geom ids that belong to the endpoint body (used for contact test)."""
        out: set[int] = set()
        if bid < 0:
            return out
        for gid in range(self.model.ngeom):
            if int(self.model.geom_bodyid[gid]) == bid:
                out.add(gid)
        return out

    def _prosthesis_hand_geoms(self) -> set[int]:
        """Palm + all finger-phalanx geoms of the MPL prosthesis (any mode
        without those bodies falls back to just the endpoint geoms)."""
        m = self.model
        hand_body_names = {
            "prosthesis/palm",
            "prosthesis/thumb0", "prosthesis/thumb1", "prosthesis/thumb2", "prosthesis/thumb3",
            "prosthesis/index0", "prosthesis/index1", "prosthesis/index2", "prosthesis/index3",
            "prosthesis/middle0", "prosthesis/middle1", "prosthesis/middle2", "prosthesis/middle3",
            "prosthesis/ring0", "prosthesis/ring1", "prosthesis/ring2", "prosthesis/ring3",
            "prosthesis/pinky0", "prosthesis/pinky1", "prosthesis/pinky2", "prosthesis/pinky3",
        }
        out: set[int] = set()
        for gid in range(m.ngeom):
            bid = int(m.geom_bodyid[gid])
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname in hand_body_names:
                out.add(gid)
        if not out:
            out = self._geoms_of_body(self._endpoint_bid)
        return out

    # --------------------------------------------------------------- state
    def raw_state(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        mlen = self.data.actuator_length.copy()
        mvel = self.data.actuator_velocity.copy()
        hand = self._hand_pos()
        return np.concatenate([qpos, qvel, mlen, mvel, hand]).astype(np.float32)

    def _hand_pos(self) -> np.ndarray:
        return self.data.xpos[self._endpoint_bid].copy()

    # -------------------------------------------------------------- reward
    def _reward(self, state: np.ndarray, action: np.ndarray) -> float:
        target = np.deg2rad(self.target_deg())
        if len(self.joint_of_interest_idx) == 0:
            return -float(np.sum(action ** 2))
        try:
            qpos_targets = np.array(
                [self.data.qpos[self.model.jnt_qposadr[j]] for j in self.joint_of_interest_idx],
                dtype=np.float64,
            )
        except Exception:
            L = self._layout
            qpos_targets = state[L.qpos][: len(target)]
        err = qpos_targets - target.astype(np.float64)
        return -(float(np.sum(err ** 2)) + 1e-3 * float(np.sum(action ** 2)))

    # -------------------------------------------------------------- helpers
    def joint_qpos(self, joint_name: str) -> float:
        j = self._joint_idx[joint_name]
        return float(self.data.qpos[self.model.jnt_qposadr[j]])

    def joint_qvel(self, joint_name: str) -> float:
        j = self._joint_idx[joint_name]
        return float(self.data.qvel[self.model.jnt_dofadr[j]])

    def close(self) -> None:
        """Release MuJoCo native resources (model + data).

        The tk GUI launches multiple bimanual scenarios in the same Python
        process; without an explicit release the previous 80-actuator model
        stays resident and the next :meth:`_build_model` hits the MuJoCo
        arena limit ("Could not allocate memory").  Calling this from the GUI
        cleanup path -- and forcing ``gc.collect()`` after -- avoids that.
        """
        for attr in ("data", "model"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    setattr(self, attr, None)
                    del obj
                except Exception:
                    pass


# --------------------------------------------------------------------- helpers
def replace_config_targets(cfg: MyoArmConfig,
                           target_deg: Tuple[float, ...]) -> MyoArmConfig:
    """Return a copy of ``cfg`` with ``target_deg_override`` set."""
    from dataclasses import replace
    return replace(cfg, target_deg_override=tuple(target_deg))
