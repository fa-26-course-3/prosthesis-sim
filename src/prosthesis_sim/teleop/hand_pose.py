"""Hand-pose representation shared between webcam trackers and pose mappers.

MediaPipe returns 21 landmarks per hand. The mapping we use in the sandbox
only needs a few *derived* quantities (wrist position, finger flex fractions,
palm plane orientation), so :class:`HandPose` stores both the raw landmarks
and the pre-computed features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# MediaPipe Hands landmark indices (21 total).
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "thumb":  (THUMB_MCP,  THUMB_IP,  THUMB_TIP),
    "index":  (INDEX_MCP,  INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring":   (RING_MCP,   RING_PIP,  RING_TIP),
    "pinky":  (PINKY_MCP,  PINKY_PIP, PINKY_TIP),
}


@dataclass
class HandPose:
    """Snapshot of one hand pose extracted from a webcam frame.

    All coordinates are already normalised so downstream mappers do not need to
    know the image size:

    * ``landmarks``  -- (21, 3) image-space, x,y in [0, 1], z relative depth.
    * ``world_landmarks`` -- (21, 3) meters, wrist-centered (MediaPipe world
      coords). May be ``None`` if the backend does not provide them.
    * ``wrist_xy`` -- palm centroid in image coords (used for shoulder yaw/pitch).
    * ``finger_flex_frac`` -- {thumb, index, middle, ring, pinky} -> [0, 1]
      where 0 = fully extended, 1 = fully curled.
    * ``wrist_flex_frac`` / ``wrist_udev_frac`` -- normalized signed [-1, +1].
    * ``hand_size_frac`` -- rough proxy for arm reach (small = far / extended).
    """

    detected: bool = False
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((21, 3), dtype=np.float32))
    world_landmarks: Optional[np.ndarray] = None
    wrist_xy: tuple[float, float] = (0.5, 0.5)
    finger_flex_frac: Dict[str, float] = field(default_factory=dict)
    wrist_flex_frac: float = 0.0
    wrist_udev_frac: float = 0.0
    hand_size_frac: float = 0.35
    t: float = 0.0

    @staticmethod
    def compute_features(landmarks: np.ndarray) -> Dict[str, float]:
        """Extract the derived features from a raw (21, 3) landmark array."""
        wrist = landmarks[WRIST]
        wrist_xy = (float(wrist[0]), float(wrist[1]))

        # Hand size ~ distance wrist -> middle MCP in the image frame.
        middle_mcp = landmarks[MIDDLE_MCP]
        hand_size_frac = float(np.linalg.norm(middle_mcp[:2] - wrist[:2]))

        # Finger flex: angle between MCP->PIP and PIP->TIP vectors.
        flex = {}
        for name, (mcp, pip, tip) in FINGERS.items():
            v1 = landmarks[pip] - landmarks[mcp]
            v2 = landmarks[tip] - landmarks[pip]
            n1 = np.linalg.norm(v1) + 1e-6
            n2 = np.linalg.norm(v2) + 1e-6
            cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            angle = float(np.arccos(cos))       # 0 = straight, pi = folded
            flex[name] = float(np.clip(angle / np.pi, 0.0, 1.0))

        # Palm plane: mainly for wrist flex/udev feedback.
        palm_up = landmarks[MIDDLE_MCP] - wrist
        palm_side = landmarks[INDEX_MCP] - landmarks[PINKY_MCP]
        # Wrist flex: how much the middle MCP is above/below the wrist in y.
        wrist_flex_frac = float(np.clip(-palm_up[1] * 2.0, -1.0, 1.0))
        # Wrist ulnar deviation: sign of the palm_side x component.
        wrist_udev_frac = float(np.clip(palm_side[0] * 2.0, -1.0, 1.0))

        return dict(
            wrist_xy=wrist_xy,
            finger_flex_frac=flex,
            wrist_flex_frac=wrist_flex_frac,
            wrist_udev_frac=wrist_udev_frac,
            hand_size_frac=hand_size_frac,
        )

    @classmethod
    def from_landmarks(cls, landmarks: np.ndarray,
                       world_landmarks: Optional[np.ndarray] = None,
                       t: float = 0.0) -> "HandPose":
        f = cls.compute_features(landmarks)
        return cls(detected=True, landmarks=landmarks.copy(),
                   world_landmarks=None if world_landmarks is None else world_landmarks.copy(),
                   t=t, **f)

    @classmethod
    def none(cls) -> "HandPose":
        return cls(detected=False)


__all__ = ["HandPose", "FINGERS", "WRIST",
           "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
           "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
           "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
           "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
           "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"]
