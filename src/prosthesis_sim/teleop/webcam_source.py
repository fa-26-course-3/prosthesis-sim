"""Webcam-driven hand pose tracker (background thread).

The tracker runs continuously so the physics loop never has to wait on a
frame.  Consumers just call :meth:`WebcamHandTracker.latest` to read the most
recent :class:`HandPose`.

Three backends are supported:

* ``mediapipe``  -- MediaPipe Hands **Tasks API** (hand only).
* ``full``       -- MediaPipe Hands **+** PoseLandmarker (hand + body) with an
                    annotated frame ready for :func:`cv2.imshow`.
* ``synthetic``  -- deterministic sinusoidal pose, no camera needed.
                    Used by CI / smoke test and by ``--no-webcam``.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .body_pose import BodyPose
from .hand_pose import (
    FINGERS, HandPose, INDEX_MCP, MIDDLE_MCP, PINKY_MCP, WRIST,
)


_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
_HAND_LANDMARKER_LOCAL = _MODELS_DIR / "hand_landmarker.task"
_POSE_LANDMARKER_LOCAL = _MODELS_DIR / "pose_landmarker.task"


def _ensure_model(url: str, local_path: Path) -> Path:
    if local_path.is_file() and local_path.stat().st_size > 0:
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[teleop] downloading MediaPipe model -> {local_path}")
    urllib.request.urlretrieve(url, str(local_path))
    return local_path


def ensure_hand_landmarker(local_path: Path = _HAND_LANDMARKER_LOCAL) -> Path:
    return _ensure_model(_HAND_LANDMARKER_URL, local_path)


def ensure_pose_landmarker(local_path: Path = _POSE_LANDMARKER_LOCAL) -> Path:
    return _ensure_model(_POSE_LANDMARKER_URL, local_path)


@dataclass
class TeleopFrame:
    """Bundle returned by the combined hand-and-body tracker."""
    hand: HandPose
    body: BodyPose
    annotated_bgr: Optional[np.ndarray] = None       # ready for cv2.imshow
    fps: float = 0.0
    t: float = 0.0


class HandTracker:
    """Protocol for anything that produces a :class:`HandPose` per call.

    Concrete implementations override ``_capture_and_infer`` and are wrapped
    by :class:`WebcamHandTracker` to run in a background thread.

    The combined ``full`` tracker additionally exposes ``latest_body()`` and
    ``latest_annotated_frame()`` so callers can show the camera panel next to
    the MuJoCo viewer.
    """

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest(self) -> HandPose: ...
    def latest_body(self) -> "BodyPose":
        return BodyPose.none()
    def latest_annotated_frame(self) -> Optional[np.ndarray]:
        return None
    def fps(self) -> float:
        return 0.0


# --------------------------------------------------------------------- base
class BaseThreadedTracker(HandTracker):
    """Common threading harness shared by MediaPipe and synthetic backends."""

    def __init__(self, target_fps: float = 30.0):
        self._target_fps = float(target_fps)
        self._lock = threading.Lock()
        self._pose: HandPose = HandPose.none()
        self._body: BodyPose = BodyPose.none()
        self._annotated: Optional[np.ndarray] = None
        self._fps_smoothed: float = 0.0
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._release()

    def latest(self) -> HandPose:
        with self._lock:
            return self._pose

    def latest_body(self) -> BodyPose:
        with self._lock:
            return self._body

    def latest_annotated_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._annotated is None else self._annotated.copy()

    def fps(self) -> float:
        with self._lock:
            return float(self._fps_smoothed)

    # ------------------------- override in subclass ------------------------
    def _step(self, t: float) -> HandPose:
        raise NotImplementedError

    def _release(self) -> None:
        pass

    # ---------------------------------------------------------------- loop
    def _loop(self) -> None:
        dt = 1.0 / max(self._target_fps, 1.0)
        t0 = time.perf_counter()
        prev = t0
        while not self._stop_evt.is_set():
            frame_start = time.perf_counter()
            try:
                self._step_and_publish(frame_start - t0)
            except Exception as e:
                with self._lock:
                    self._pose = HandPose.none()
                    self._body = BodyPose.none()
                print(f"[HandTracker] frame error: {e!r}")
            # Frame-rate cap and FPS estimate.
            elapsed = time.perf_counter() - frame_start
            inst_fps = 1.0 / max(time.perf_counter() - prev, 1e-3)
            prev = time.perf_counter()
            with self._lock:
                self._fps_smoothed = 0.9 * self._fps_smoothed + 0.1 * inst_fps
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # ------------------ default publisher: hand-only backends ---------------
    def _step_and_publish(self, t: float) -> None:
        pose = self._step(t)
        with self._lock:
            self._pose = pose


# ----------------------------------------------------------------- synthetic
class SyntheticHandTracker(BaseThreadedTracker):
    """No-camera tracker: emits a deterministic breathing / waving pose.

    Useful for CI, headless smoke tests, and for demo-ing the teleop pipeline
    on a laptop without a webcam.
    """

    def __init__(self, target_fps: float = 30.0, seed: int = 0):
        super().__init__(target_fps=target_fps)
        self._rng = np.random.default_rng(seed)

    def _step(self, t: float) -> HandPose:
        # Build a synthetic set of 21 landmarks that wave and flex.
        base = np.zeros((21, 3), dtype=np.float32)
        # Wrist follows a slow horizontal figure-8 around the image centre.
        base[WRIST] = [0.5 + 0.15 * np.sin(0.6 * t),
                       0.5 + 0.08 * np.sin(1.2 * t),
                       0.0]
        # MCPs sit above the wrist.
        for k, mcp in enumerate((INDEX_MCP, MIDDLE_MCP, 13, PINKY_MCP)):
            base[mcp] = base[WRIST] + [0.05 * (k - 1.5), -0.08, 0.0]
        base[1] = base[WRIST] + [-0.06, -0.04, 0.0]
        base[2] = base[1] + [-0.03, -0.02, 0.0]
        # Fingers alternately curl every ~3 s.
        flex_scale = 0.5 * (1.0 + np.sin(0.8 * t))
        finger_names = ("thumb", "index", "middle", "ring", "pinky")
        for name in finger_names:
            mcp, pip, tip = FINGERS[name]
            straight = base[mcp] + [0.0, -0.06, 0.0]
            curled = base[mcp] + [0.02, +0.01, 0.0]
            base[pip] = (1 - flex_scale) * straight + flex_scale * curled
            straight2 = base[pip] + [0.0, -0.04, 0.0]
            curled2 = base[pip] + [0.03, +0.02, 0.0]
            base[tip] = (1 - flex_scale) * straight2 + flex_scale * curled2
        # DIPs mid-way (not used for our features but keeps arrays populated).
        for finger in ("index", "middle", "ring", "pinky"):
            mcp, pip, tip = FINGERS[finger]
            base[pip + 1] = 0.5 * (base[pip] + base[tip])
        return HandPose.from_landmarks(base, world_landmarks=None, t=t)

    def _step_and_publish(self, t: float) -> None:
        hand = self._step(t)
        # Also synthesize a body pose that "waves" the arm around the target.
        body_landmarks = np.zeros((33, 4), dtype=np.float32)
        body_landmarks[:, 3] = 0.9        # visibility
        # Shoulder fixed.
        S = np.array([0.35, 0.25, 0.0, 0.9], dtype=np.float32)
        H = np.array([0.35, 0.75, 0.0, 0.9], dtype=np.float32)
        # Elbow / wrist oscillate to exercise the mapper.
        elbow_bend = 0.5 * (1 + np.sin(0.4 * t))            # 0 -> 1
        arm_side   = 0.15 * np.sin(0.3 * t)
        E = S + np.array([arm_side, 0.20 + 0.05 * elbow_bend, 0.0, 0.0],
                         dtype=np.float32)
        W = E + np.array([arm_side * 0.5, 0.18 * (0.4 + 0.6 * elbow_bend), 0.0, 0.0],
                         dtype=np.float32)
        body_landmarks[11] = S
        body_landmarks[13] = E
        body_landmarks[15] = W
        body_landmarks[23] = H
        body = BodyPose.from_landmarks(body_landmarks, world_landmarks=None, t=t)
        with self._lock:
            self._pose = hand
            self._body = body
            self._annotated = None      # no real frame in synthetic mode


# ---------------------------------------------------------------- MediaPipe
class WebcamHandTracker(BaseThreadedTracker):
    """MediaPipe Tasks HandLandmarker over cv2.VideoCapture.

    Uses the new ``mediapipe.tasks.python.vision.HandLandmarker`` (the
    legacy ``mediapipe.solutions.hands`` module was removed in MediaPipe 1.0).
    """

    def __init__(self, camera_index: int = 0, target_fps: float = 30.0,
                 max_hands: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 mirror: bool = True,
                 model_path: Optional[str] = None):
        super().__init__(target_fps=target_fps)
        import cv2                                       # noqa: F401
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode,
        )
        self._cv2 = cv2
        self._mp = mp
        self._mirror = bool(mirror)
        self._cap = cv2.VideoCapture(int(camera_index))
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"Could not open camera index {camera_index}. "
                "Try --no-webcam for the synthetic backend."
            )
        self._cap.set(cv2.CAP_PROP_FPS, target_fps)

        task_path = Path(model_path) if model_path else ensure_hand_landmarker()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(task_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=int(max_hands),
            min_hand_detection_confidence=float(min_detection_confidence),
            min_hand_presence_confidence=float(min_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def _release(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        try:
            self._landmarker.close()
        except Exception:
            pass

    def _step(self, t: float) -> HandPose:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return HandPose.none()
        if self._mirror:
            frame = self._cv2.flip(frame, 1)
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(t * 1000.0)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not result.hand_landmarks:
            return HandPose.none()
        lm = result.hand_landmarks[0]
        arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        world = None
        if result.hand_world_landmarks:
            world = np.array(
                [[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]],
                dtype=np.float32,
            )
        return HandPose.from_landmarks(arr, world_landmarks=world, t=t)


# ------------------------------------- combined hand + body + annotated frame
class WebcamHandAndBodyTracker(BaseThreadedTracker):
    """MediaPipe HandLandmarker **+** PoseLandmarker with an annotated frame.

    Publishes:

    * ``latest()``          -- the :class:`HandPose` (as any other tracker),
    * ``latest_body()``     -- the :class:`BodyPose` (shoulder / elbow / wrist),
    * ``latest_annotated_frame()`` -- BGR image with markers, colored lines
      and per-joint angles baked in, ready for ``cv2.imshow``.
    """

    def __init__(self, camera_index: int = 0, target_fps: float = 30.0,
                 max_hands: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 mirror: bool = True,
                 hand_model_path: Optional[str] = None,
                 pose_model_path: Optional[str] = None,
                 draw_overlay: bool = True):
        super().__init__(target_fps=target_fps)
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions,
            PoseLandmarker, PoseLandmarkerOptions,
            RunningMode,
        )
        self._cv2 = cv2
        self._mp = mp
        self._mirror = bool(mirror)
        self._draw_overlay = bool(draw_overlay)
        self._cap = cv2.VideoCapture(int(camera_index))
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"Could not open camera index {camera_index}. "
                "Try --no-webcam for the synthetic backend."
            )
        self._cap.set(cv2.CAP_PROP_FPS, target_fps)

        hand_path = Path(hand_model_path) if hand_model_path else ensure_hand_landmarker()
        pose_path = Path(pose_model_path) if pose_model_path else ensure_pose_landmarker()

        self._hand = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=int(max_hands),
            min_hand_detection_confidence=float(min_detection_confidence),
            min_hand_presence_confidence=float(min_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        ))
        self._pose_lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=float(min_detection_confidence),
            min_pose_presence_confidence=float(min_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        ))

    def _release(self) -> None:
        for obj_attr in ("_cap", "_hand", "_pose_lm"):
            obj = getattr(self, obj_attr, None)
            if obj is not None:
                try:
                    if obj_attr == "_cap":
                        obj.release()
                    else:
                        obj.close()
                except Exception:
                    pass

    def _step_and_publish(self, t: float) -> None:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            with self._lock:
                self._pose = HandPose.none()
                self._body = BodyPose.none()
            return
        if self._mirror:
            frame = self._cv2.flip(frame, 1)
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(t * 1000.0)

        # --- Hand ---
        try:
            hand_result = self._hand.detect_for_video(mp_image, ts_ms)
        except Exception:
            hand_result = None
        hand_pose = HandPose.none()
        if hand_result is not None and hand_result.hand_landmarks:
            lm = hand_result.hand_landmarks[0]
            arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            world = None
            if hand_result.hand_world_landmarks:
                world = np.array(
                    [[p.x, p.y, p.z] for p in hand_result.hand_world_landmarks[0]],
                    dtype=np.float32,
                )
            hand_pose = HandPose.from_landmarks(arr, world_landmarks=world, t=t)

        # --- Body ---
        try:
            pose_result = self._pose_lm.detect_for_video(mp_image, ts_ms)
        except Exception:
            pose_result = None
        body_pose = BodyPose.none()
        if pose_result is not None and pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks[0]
            body_arr = np.array(
                [[p.x, p.y, p.z, p.visibility] for p in lm],
                dtype=np.float32,
            )
            world = None
            if getattr(pose_result, "pose_world_landmarks", None):
                world = np.array(
                    [[p.x, p.y, p.z, p.visibility] for p in pose_result.pose_world_landmarks[0]],
                    dtype=np.float32,
                )
            body_pose = BodyPose.from_landmarks(body_arr, world_landmarks=world, t=t)

        annotated = None
        if self._draw_overlay:
            # Late-import so `import prosthesis_sim.teleop` doesn't drag cv2 in
            # for callers that don't need the overlay.
            from .viewer import draw_body, draw_hand, draw_hud
            annotated = frame.copy()
            draw_body(annotated, body_pose)
            draw_hand(annotated, hand_pose)
            draw_hud(annotated, body_pose, hand_pose, fps=self._fps_smoothed)

        with self._lock:
            self._pose = hand_pose
            self._body = body_pose
            self._annotated = annotated


def make_tracker(kind: str = "mediapipe", **kw) -> HandTracker:
    """Factory used by scripts and the GUI.

    * ``mediapipe`` / ``hand``  -- MediaPipe HandLandmarker only.
    * ``full`` / ``body``       -- HandLandmarker + PoseLandmarker + annotated
                                   frame (used by the dual-view script).
    * ``synthetic``             -- deterministic no-camera fallback.

    Any camera or MediaPipe failure falls back to the synthetic backend so demo
    scripts always start.
    """
    kind = kind.lower()
    if kind in ("synthetic", "none", "off"):
        return SyntheticHandTracker(**{k: v for k, v in kw.items()
                                       if k in ("target_fps", "seed")})
    try:
        if kind in ("full", "body", "hand_and_body", "combined"):
            return WebcamHandAndBodyTracker(**kw)
        return WebcamHandTracker(**kw)
    except Exception as e:
        print(f"[teleop] webcam init failed ({e!r}); falling back to synthetic tracker.")
        return SyntheticHandTracker(target_fps=kw.get("target_fps", 30.0))
