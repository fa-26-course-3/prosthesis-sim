"""Map a :class:`HandPose` (and optional :class:`BodyPose`) into MPL prosthesis
joint-angle targets.

Everything is expressed as a ``{actuator_name: target_rad}`` dict. The
:class:`WebcamTeleopController` then converts these into an action vector
using the position-servo pairs the env exposes.

Two mapping modes:

* **hand-only**  -- shoulder / elbow driven from wrist-in-image proxies (used
                    when body tracking is not available).
* **body-aware** -- shoulder / elbow driven from *real* body angles measured
                    by PoseLandmarker: the shoulder flex angle, the elbow bend
                    angle, and a 2D abduction proxy from the arm's horizontal
                    offset.  Falls back to the hand-only path if body isn't
                    detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .body_pose import BodyPose
from .hand_pose import HandPose


@dataclass
class MappingConfig:
    """Normalised gains that scale hand-pose features into servo ctrl range."""

    # Shoulder / elbow ranges are read from the MPL XML ctrl range (rad):
    #   shoulder_fe    in [-0.611, +2.792]   (positive = flex forward)
    #   shoulder_abad  in [-2.792, +0.000]   (negative = abduct outward)
    #   elbow          in [+0.050, +2.356]
    # We map each pose feature to a subrange within those bounds.
    shoulder_fe_min: float  = -0.10
    shoulder_fe_max: float  = +1.80
    shoulder_abad_min: float = -1.60
    shoulder_abad_max: float =  0.00
    elbow_min: float = 0.30
    elbow_max: float = 1.80
    # Wrist DOFs.
    wrist_flex_max: float = 0.8   # rad, ±
    wrist_udev_max: float = 0.4
    # Finger MCP ctrl range top (radians). MPL A_*_MCP: [0, 1.6] etc.
    finger_mcp_max: float = 1.5
    thumb_mcp_max: float = 1.0
    # Low-pass alpha for smoothing (0..1).  Higher = more responsive.
    smoothing_alpha: float = 0.35


class PoseMapper:
    """Stateful mapper: applies EMA smoothing so servos don't twitch."""

    def __init__(self, cfg: MappingConfig | None = None):
        self.cfg = cfg or MappingConfig()
        self._last: Dict[str, float] = {}

    def reset(self) -> None:
        self._last.clear()

    def map(self, pose: HandPose,
            body: Optional[BodyPose] = None) -> Dict[str, float]:
        """Turn a hand + (optional) body pose into ``{joint_name -> target_rad}``.

        Body-aware mapping wins over the hand-only fallback whenever the body
        landmarks are visible; if either channel drops out we hold the last
        commanded value so servos never twitch.
        """
        if not pose.detected and (body is None or not body.detected):
            return dict(self._last)
        c = self.cfg
        raw: Dict[str, float] = {}

        # ---------------- shoulder / elbow -----------------------------------
        if body is not None and body.detected:
            # body.shoulder_flex_deg  in ~[10, 180] deg (180 = arm down),
            # body.elbow_angle_deg    in ~[0, 180]   deg (180 = elbow straight).
            # Map arm-down / straight -> MPL neutral, arm-up / bent -> MPL flexed.
            shoulder_fe_deg = _clip(180.0 - body.shoulder_flex_deg, 0.0, 180.0)
            shoulder_fe = np.deg2rad(shoulder_fe_deg) * (c.shoulder_fe_max / (np.pi))
            shoulder_fe = _lerp(c.shoulder_fe_min, c.shoulder_fe_max,
                                shoulder_fe_deg / 180.0)
            elbow_bend_deg = _clip(180.0 - body.elbow_angle_deg, 0.0, 180.0)
            elbow = _lerp(c.elbow_min, c.elbow_max, elbow_bend_deg / 130.0)
            # Abduction: 0 deg -> arm along body, ~90 deg -> arm out to the side.
            shoulder_abad = _lerp(0.0, c.shoulder_abad_min,
                                  body.shoulder_abad_deg / 60.0)
        elif pose.detected:
            xy = pose.wrist_xy
            shoulder_fe   = _lerp(c.shoulder_fe_min, c.shoulder_fe_max, 1.0 - xy[1])
            shoulder_abad = _lerp(c.shoulder_abad_min, c.shoulder_abad_max, xy[0])
            hs = _clip((pose.hand_size_frac - 0.10) / (0.35 - 0.10), 0.0, 1.0)
            elbow = _lerp(c.elbow_min, c.elbow_max, hs)
        else:
            shoulder_fe = 0.0
            shoulder_abad = -0.6
            elbow = 0.9

        raw["prosthesis/Lshoulder_fe"]   = shoulder_fe
        raw["prosthesis/Lshoulder_abad"] = shoulder_abad
        raw["prosthesis/Lhumeral_rot"]   = 0.0
        raw["prosthesis/Lelbow"]         = elbow

        # ---------------- wrist / fingers (hand-only) ------------------------
        if pose.detected:
            f = pose.finger_flex_frac
            raw["prosthesis/A_wrist_FLEX"] = c.wrist_flex_max * float(pose.wrist_flex_frac)
            raw["prosthesis/A_wrist_UDEV"] = c.wrist_udev_max * float(pose.wrist_udev_frac)
            raw["prosthesis/A_wrist_PRO"]  = 0.0
            raw["prosthesis/A_thumb_MCP"]  = c.thumb_mcp_max  * f.get("thumb",  0.0)
            raw["prosthesis/A_thumb_ABD"]  = 0.3 * f.get("thumb", 0.0)
            raw["prosthesis/A_thumb_PIP"]  = 0.6 * f.get("thumb", 0.0)
            raw["prosthesis/A_thumb_DIP"]  = 0.5 * f.get("thumb", 0.0)
            raw["prosthesis/A_index_MCP"]  = c.finger_mcp_max * f.get("index",  0.0)
            raw["prosthesis/A_index_ABD"]  = 0.1
            raw["prosthesis/A_middle_MCP"] = c.finger_mcp_max * f.get("middle", 0.0)
            raw["prosthesis/A_ring_MCP"]   = c.finger_mcp_max * f.get("ring",   0.0)
            raw["prosthesis/A_pinky_MCP"]  = c.finger_mcp_max * f.get("pinky",  0.0)
            raw["prosthesis/A_pinky_ABD"]  = 0.1

        # EMA smoothing.
        a = float(c.smoothing_alpha)
        for k, v in raw.items():
            self._last[k] = a * v + (1.0 - a) * self._last.get(k, v)
        return dict(self._last)


def _lerp(lo: float, hi: float, t: float) -> float:
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return lo + (hi - lo) * t


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


__all__ = ["MappingConfig", "PoseMapper"]
