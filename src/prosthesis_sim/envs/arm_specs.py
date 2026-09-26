"""Arm mode registry.

Each entry describes a MuJoCo/MyoSuite model + the metadata the rest of the
stack needs to be *mode-agnostic* (which joints are of interest, which
actuators are biological muscles vs prosthetic motors, where the endpoint
body is, and so on).

Add a new mode = add one :class:`ArmSpec` to :data:`MODES`; every module
(env, sensory pipeline, controllers, GUI) will pick it up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import myosuite  # noqa: F401  (installs the simhive path)


_PKG_ROOT = os.path.dirname(myosuite.__file__)
_XML_SEARCH_ROOTS = (
    os.path.join(_PKG_ROOT, "simhive/myo_sim"),
    os.path.join(_PKG_ROOT, "simhive"),
    _PKG_ROOT,                     # allows "envs/myo/assets/..." paths
)


@dataclass(frozen=True)
class ArmSpec:
    """Static metadata describing a single arm mode."""

    mode: str
    xml_path: str
    joints_of_interest: Tuple[str, ...]           # controllable joints we care about
    joint_targets_deg: Tuple[float, ...]          # per-joint reach target (same order as joints_of_interest)
    endpoint_body: str                             # body used for hand xyz
    ball_parent_body: Optional[str]               # where to attach a hand-held mass, or None
    # Actuator categorisation (by name; indices are resolved at env build time)
    flexor_actuators: Tuple[str, ...]             # flex the primary target joint (usually elbow)
    extensor_actuators: Tuple[str, ...]           # extend the primary target joint
    prosthetic_actuators: Tuple[str, ...]         # non-biological motors (empty for pure human)
    biological_actuators: Tuple[str, ...]         # everything else (usually all muscles)
    # Optional mapping (joint name -> prosthesis actuator name) for MPL-style
    # position-servo prostheses: PD controller sends target angles directly.
    prosth_position_actuators: Tuple[Tuple[str, str], ...] = ()
    is_prosthesis: bool = False
    description: str = ""

    def resolve_xml(self) -> str:
        if os.path.isabs(self.xml_path):
            return self.xml_path
        for root in _XML_SEARCH_ROOTS:
            candidate = os.path.join(root, self.xml_path)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"Cannot locate XML '{self.xml_path}' under any of {_XML_SEARCH_ROOTS}"
        )


# ---------------------------------------------------------------- registry --
MODES: dict[str, ArmSpec] = {
    # -------- HUMAN mode --------
    "human_elbow": ArmSpec(
        mode="human_elbow",
        xml_path="elbow/myoelbow_2dof6muscles.xml",
        joints_of_interest=("r_shoulder_elev", "r_elbow_flex"),
        joint_targets_deg=(0.0, 90.0),
        endpoint_body="r_ulna_radius_hand",
        ball_parent_body="r_ulna_radius_hand",
        flexor_actuators=("BIClong", "BICshort", "BRA"),
        extensor_actuators=("TRIlong", "TRIlat", "TRImed"),
        prosthetic_actuators=(),
        biological_actuators=("TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"),
        is_prosthesis=False,
        description="Biological 2-joint, 6-muscle elbow (MyoSuite default demo model).",
    ),

    # -------- PROSTHESIS mode (MyoChallenge-2024 bimanual model) --------
    "prosthesis_arm": ArmSpec(
        mode="prosthesis_arm",
        # Full myoArm + Modular Prosthetic Limb (MPL) bimanual model used by
        # MyoChallenge-2024 "Prosthesis Co-Manipulation".
        xml_path="envs/myo/assets/arm/myoarm_bionic_bimanual.xml",
        # Four MPL arm joints are the "joints of interest" -- exposing all of
        # them lets the controller move the hand in every direction (up, down,
        # forward, sideways, across the body), not just flex the elbow.
        joints_of_interest=(
            "prosthesis/Lshoulder_fe",
            "prosthesis/Lshoulder_abad",
            "prosthesis/Lhumeral_rot",
            "prosthesis/Lelbow",
        ),
        # Default "reach forward" pose -- see TARGET_POSES for named alternatives.
        joint_targets_deg=(45.0, -30.0, 0.0, 90.0),
        endpoint_body="prosthesis/palm",
        ball_parent_body="prosthesis/palm",
        # No agonist/antagonist muscle groups drive the prosthesis: the whole
        # left arm is servo-controlled. Empty flexor/extensor tuples signal
        # the PD controller to use the position-servo path below.
        flexor_actuators=(),
        extensor_actuators=(),
        prosthetic_actuators=(
            "prosthesis/Lshoulder_fe", "prosthesis/Lshoulder_abad",
            "prosthesis/Lhumeral_rot", "prosthesis/Lelbow",
            "prosthesis/A_wrist_PRO", "prosthesis/A_wrist_UDEV", "prosthesis/A_wrist_FLEX",
            "prosthesis/A_thumb_ABD", "prosthesis/A_thumb_MCP",
            "prosthesis/A_thumb_PIP", "prosthesis/A_thumb_DIP",
            "prosthesis/A_index_ABD", "prosthesis/A_index_MCP",
            "prosthesis/A_middle_MCP", "prosthesis/A_ring_MCP",
            "prosthesis/A_pinky_ABD", "prosthesis/A_pinky_MCP",
        ),
        # 63 myoArm muscles held at a small resting tone so the biological
        # right arm stays roughly in its neutral MuJoCo home pose while the
        # left MPL prosthesis is what the controller drives.
        biological_actuators=(
            "DELT1", "DELT2", "DELT3", "SUPSP", "INFSP", "SUBSC", "TMIN", "TMAJ",
            "PECM1", "PECM2", "PECM3", "LAT1", "LAT2", "LAT3", "CORB",
            "TRIlong", "TRIlat", "TRImed", "ANC", "SUP",
            "BIClong", "BICshort", "BRA", "BRD",
            "ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ",
            "FDS5", "FDS4", "FDS3", "FDS2",
            "FDP5", "FDP4", "FDP3", "FDP2",
            "EDC5", "EDC4", "EDC3", "EDC2", "EDM", "EIP",
            "EPL", "EPB", "FPL", "APL", "OP",
            "RI2", "LU_RB2", "UI_UB2",
            "RI3", "LU_RB3", "UI_UB3",
            "RI4", "LU_RB4", "UI_UB4",
            "RI5", "LU_RB5", "UI_UB5",
        ),
        # Position-servo mapping: (joint name  ->  actuator name).  The PD
        # controller writes the *target angle* straight into these actuators
        # because MPL joints are position-controlled per MyoChallenge-2024.
        prosth_position_actuators=(
            ("prosthesis/Lshoulder_fe",   "prosthesis/Lshoulder_fe"),
            ("prosthesis/Lshoulder_abad", "prosthesis/Lshoulder_abad"),
            ("prosthesis/Lhumeral_rot",   "prosthesis/Lhumeral_rot"),
            ("prosthesis/Lelbow",         "prosthesis/Lelbow"),
        ),
        is_prosthesis=True,
        description="MyoArm + Modular Prosthetic Limb (MPL, 17 servos) -- "
                    "4 arm servos exposed for full-3D reach; MyoChallenge-2024 style.",
    ),
}


# Named target poses for prosthesis_arm.  Each tuple matches
# ``prosthesis_arm.joints_of_interest`` order:
#   (shoulder_fe, shoulder_abad, humeral_rot, elbow)  in degrees.
# Sign conventions come from the MPL XML ctrl ranges:
#   shoulder_fe    in [-35, +160]   (positive = flex forward / up)
#   shoulder_abad  in [-160, 0]     (negative = abduct outward)
#   humeral_rot    in [-30, +85]
#   elbow          in [+3, +135]    (positive = flex)
TARGET_POSES: dict[str, Tuple[float, float, float, float]] = {
    "neutral":       (  0.0,   0.0,  0.0,   3.0),   # arm hanging down
    "reach_forward": ( 45.0, -30.0,  0.0,  90.0),
    "reach_up":      (120.0, -20.0,  0.0,  20.0),
    "reach_across":  ( 45.0, -90.0, 30.0,  90.0),   # across the body
    "reach_side":    ( 20.0,-120.0,  0.0,  60.0),   # laterally outward
    "reach_down":    (-30.0, -20.0,  0.0,  30.0),
}


DEFAULT_MODE = "human_elbow"


def get_spec(mode: str) -> ArmSpec:
    if mode not in MODES:
        raise KeyError(
            f"unknown arm mode '{mode}'. Available: {sorted(MODES)}"
        )
    return MODES[mode]


def list_modes() -> list[str]:
    return list(MODES.keys())
