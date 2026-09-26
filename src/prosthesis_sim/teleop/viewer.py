"""Drawing helpers for the teleop camera panel.

Given a BGR frame plus the latest :class:`BodyPose` / :class:`HandPose`, we
paint:

* one colored marker per tracked joint (shoulder, elbow, wrist),
* colored lines connecting them,
* a small HUD with real-time joint angles + hand-flex fractions,
* thin skeleton lines for the finger landmarks.

Kept dependency-free (only numpy + opencv) so the whole teleop stack still
works if PoseLandmarker is unavailable -- we just skip the body overlay.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .body_pose import BodyPose
from .hand_pose import (
    FINGERS, HandPose,
    INDEX_MCP, MIDDLE_MCP, PINKY_MCP, RING_MCP, WRIST,
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP,
)


# BGR colors (OpenCV convention).
COL_SHOULDER = (60, 220, 60)     # green
COL_ELBOW    = (0, 165, 255)     # orange
COL_WRIST    = (255, 90, 90)     # blue
COL_UPPER    = (60, 220, 60)     # shoulder -> elbow line (green)
COL_FOREARM  = (0, 165, 255)     # elbow    -> wrist line (orange)
COL_HAND     = (255, 90, 90)     # wrist skeleton
COL_HUD_BG   = (20, 20, 20)
COL_HUD_FG   = (240, 240, 240)


def _px(pt: Tuple[float, float], w: int, h: int) -> Tuple[int, int]:
    return int(pt[0] * w), int(pt[1] * h)


def draw_body(frame: np.ndarray, body: BodyPose,
              show_labels: bool = True) -> None:
    """Paint shoulder/elbow/wrist markers + connection lines onto ``frame``."""
    if not body.detected:
        return
    h, w = frame.shape[:2]
    import cv2
    S = _px(body.shoulder_xy, w, h)
    E = _px(body.elbow_xy, w, h)
    W = _px(body.wrist_xy, w, h)

    # Bones first (so joints draw on top).
    cv2.line(frame, S, E, COL_UPPER,   thickness=4, lineType=cv2.LINE_AA)
    cv2.line(frame, E, W, COL_FOREARM, thickness=4, lineType=cv2.LINE_AA)

    # Joints.
    cv2.circle(frame, S, 9, COL_SHOULDER, -1, cv2.LINE_AA)
    cv2.circle(frame, E, 9, COL_ELBOW,    -1, cv2.LINE_AA)
    cv2.circle(frame, W, 9, COL_WRIST,    -1, cv2.LINE_AA)

    if show_labels:
        cv2.putText(frame, f"{body.shoulder_flex_deg:5.1f} deg",
                    (S[0] + 12, S[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_SHOULDER, 2, cv2.LINE_AA)
        cv2.putText(frame, f"elbow {body.elbow_angle_deg:5.1f} deg",
                    (E[0] + 12, E[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_ELBOW, 2, cv2.LINE_AA)
        cv2.putText(frame, f"wrist",
                    (W[0] + 12, W[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_WRIST, 2, cv2.LINE_AA)


# Finger connection topology (parent index, child index).
_FINGER_CHAINS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
]
for _mcp, _pip, _tip in FINGERS.values():
    _FINGER_CHAINS.append((WRIST, _mcp))
    _FINGER_CHAINS.append((_mcp, _pip))
    # DIP sits at index pip+1 for the 4 fingers (thumb handled above).
    if _mcp not in (THUMB_MCP,):
        _FINGER_CHAINS.append((_pip, _pip + 1))
        _FINGER_CHAINS.append((_pip + 1, _tip))


def draw_hand(frame: np.ndarray, hand: HandPose, thickness: int = 2) -> None:
    """Thin skeleton overlay for the tracked hand (fingertip lines)."""
    if not hand.detected:
        return
    h, w = frame.shape[:2]
    import cv2
    lm = hand.landmarks
    for a, b in _FINGER_CHAINS:
        if a >= len(lm) or b >= len(lm):
            continue
        pa = _px((lm[a][0], lm[a][1]), w, h)
        pb = _px((lm[b][0], lm[b][1]), w, h)
        cv2.line(frame, pa, pb, COL_HAND, thickness=thickness, lineType=cv2.LINE_AA)
    for i in range(len(lm)):
        p = _px((lm[i][0], lm[i][1]), w, h)
        cv2.circle(frame, p, 3, COL_HAND, -1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, body: BodyPose, hand: HandPose,
             fps: Optional[float] = None) -> None:
    """Small HUD panel top-left with real-time joint angles + finger flex."""
    import cv2
    h, w = frame.shape[:2]
    # Semi-transparent panel.
    panel_w, panel_h = 270, 185
    x0, y0 = 8, 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), COL_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    def _line(y, text, color=COL_HUD_FG):
        cv2.putText(frame, text, (x0 + 10, y0 + y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    _line(22, "Real arm -> MPL prosthesis")
    _line(45, f"shoulder flex : {body.shoulder_flex_deg:6.1f} deg",
          COL_SHOULDER if body.detected else (120, 120, 120))
    _line(65, f"shoulder abad : {body.shoulder_abad_deg:6.1f} deg",
          COL_SHOULDER if body.detected else (120, 120, 120))
    _line(85, f"elbow bend    : {body.elbow_angle_deg:6.1f} deg",
          COL_ELBOW if body.detected else (120, 120, 120))
    if hand.detected:
        f = hand.finger_flex_frac
        _line(110, f"thumb  {f.get('thumb', 0):.2f}   index {f.get('index', 0):.2f}",
              COL_WRIST)
        _line(128, f"middle {f.get('middle', 0):.2f}   ring  {f.get('ring', 0):.2f}",
              COL_WRIST)
        _line(146, f"pinky  {f.get('pinky', 0):.2f}",
              COL_WRIST)
    else:
        _line(120, "hand:  no detection", (120, 120, 120))
    if fps is not None:
        _line(170, f"tracker fps: {fps:5.1f}")


__all__ = ["draw_body", "draw_hand", "draw_hud"]
