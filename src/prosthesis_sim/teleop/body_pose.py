"""Full-body pose representation (shoulder + elbow + wrist + joint angles).

Complements :class:`HandPose`. Body landmarks come from MediaPipe Tasks
``PoseLandmarker`` (33 landmarks).  The camera feed is mirrored *before*
inference so the user's *right* physical arm appears on the model's *left*
side; we therefore track ``LEFT_*`` indices to control the MPL prosthesis
(which lives on the arm's left side in the bimanual XML).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# MediaPipe Pose (BlazePose) landmark indices (33 total). We only need
# upper-body reference points to derive arm joint angles.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24


@dataclass
class BodyPose:
    """Snapshot of a body pose used to drive the MPL arm.

    All 2D coordinates are normalized to the frame ([0, 1]).
    """

    detected: bool = False
    # Full 33 landmark array in image space (x, y, z, visibility).
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((33, 4), dtype=np.float32))
    # 33 world-space landmarks in meters (hip-centered).  May be None.
    world_landmarks: Optional[np.ndarray] = None

    # --------- derived (in the mirrored image, so LEFT_* == user's right) ---
    shoulder_xy: Tuple[float, float] = (0.35, 0.30)
    elbow_xy: Tuple[float, float]    = (0.35, 0.55)
    wrist_xy: Tuple[float, float]    = (0.35, 0.80)
    hip_xy: Tuple[float, float]      = (0.40, 0.75)

    # Angles in degrees (for on-screen display + control mapping).
    elbow_angle_deg: float    = 180.0     # 180 = fully extended arm
    shoulder_flex_deg: float  = 180.0     # 180 = arm hanging along the body
    shoulder_abad_deg: float  = 0.0       # 0 = arm along body, positive = abducted outward

    visibility: float = 0.0                # minimum visibility of S/E/W
    t: float = 0.0

    @staticmethod
    def compute_angles(shoulder: np.ndarray, elbow: np.ndarray,
                       wrist: np.ndarray, hip: np.ndarray
                       ) -> Tuple[float, float, float]:
        """Return (elbow, shoulder_flex, shoulder_abad) in degrees, image plane."""
        elbow_deg = _angle(shoulder[:2], elbow[:2], wrist[:2])
        shoulder_flex_deg = _angle(hip[:2], shoulder[:2], elbow[:2])
        # 2D proxy for abduction: horizontal offset of elbow from shoulder,
        # normalized by shoulder-hip distance.
        torso_len = np.linalg.norm(shoulder[:2] - hip[:2]) + 1e-6
        dx = elbow[0] - shoulder[0]
        # Positive dx (in mirrored frame the user's right arm going outward
        # means their MPL should abduct); we return degrees in [0, 90].
        shoulder_abad_deg = float(np.rad2deg(np.arctan2(abs(dx), torso_len)))
        return elbow_deg, shoulder_flex_deg, shoulder_abad_deg

    @classmethod
    def from_landmarks(cls, landmarks: np.ndarray,
                       world_landmarks: Optional[np.ndarray] = None,
                       t: float = 0.0) -> "BodyPose":
        if landmarks.shape != (33, 4):
            return cls.none()
        S = landmarks[LEFT_SHOULDER]
        E = landmarks[LEFT_ELBOW]
        W = landmarks[LEFT_WRIST]
        H = landmarks[LEFT_HIP]
        vis = float(min(S[3], E[3], W[3]))
        elbow_deg, shoulder_flex_deg, shoulder_abad_deg = cls.compute_angles(S, E, W, H)
        return cls(
            detected=True,
            landmarks=landmarks.copy(),
            world_landmarks=None if world_landmarks is None else world_landmarks.copy(),
            shoulder_xy=(float(S[0]), float(S[1])),
            elbow_xy=(float(E[0]), float(E[1])),
            wrist_xy=(float(W[0]), float(W[1])),
            hip_xy=(float(H[0]), float(H[1])),
            elbow_angle_deg=elbow_deg,
            shoulder_flex_deg=shoulder_flex_deg,
            shoulder_abad_deg=shoulder_abad_deg,
            visibility=vis,
            t=t,
        )

    @classmethod
    def none(cls) -> "BodyPose":
        return cls(detected=False)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex ``b`` between rays b->a and b->c, in degrees."""
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    v2 = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


__all__ = ["BodyPose", "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "LEFT_HIP",
           "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_HIP"]
