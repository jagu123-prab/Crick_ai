"""
Thin wrapper around MediaPipe's Pose Landmarker (Tasks API).

On first run this will download the pose landmark model (~5-30MB depending
on which variant you pick) into ./models/. That requires an internet
connection once; after that it's cached locally.
"""
import os
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# "lite" = fastest / least accurate, "full" = balanced, "heavy" = most accurate / slowest
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def _ensure_model(variant: str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"pose_landmarker_{variant}.task")
    if not os.path.exists(path):
        url = MODEL_URLS[variant]
        print(f"[pose_estimator] Downloading model ({variant}) from {url} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[pose_estimator] Saved to {path}")
    return path


class PoseEstimator:
    """
    Runs pose estimation over a video, frame by frame, in VIDEO mode
    (which lets MediaPipe use temporal smoothing between frames).
    """

    def __init__(self, model_variant: str = "full", min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        model_path = _ensure_model(model_variant)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int):
        """
        Returns a (33, 4) numpy array of [x_px, y_px, z, visibility] for the
        first detected person, or None if nobody was detected in this frame.
        """
        rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        landmarks = result.pose_landmarks[0]  # first detected person
        out = np.zeros((len(landmarks), 4), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            out[i] = [lm.x * w, lm.y * h, lm.z, lm.visibility]
        return out

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
