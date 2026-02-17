import csv
import json
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from domain.models import LandmarksGenerationResult
from infrastructure.model_downloader import ensure_model_exists

TOTAL_POSE_LANDMARKS = 33


def _landmark_to_dict(landmark: Any) -> dict[str, float | None]:
    return {
        "x": float(getattr(landmark, "x", 0.0)),
        "y": float(getattr(landmark, "y", 0.0)),
        "z": float(getattr(landmark, "z", 0.0)),
        "visibility": (
            float(landmark.visibility)
            if getattr(landmark, "visibility", None) is not None
            else None
        ),
        "presence": (
            float(landmark.presence)
            if getattr(landmark, "presence", None) is not None
            else None
        ),
    }


def _csv_fieldnames(total_landmarks: int = TOTAL_POSE_LANDMARKS) -> list[str]:
    fieldnames = ["frame", "timestamp_ms", "pose_detected"]
    for idx in range(total_landmarks):
        fieldnames.extend(
            [
                f"lm_{idx}_x",
                f"lm_{idx}_y",
                f"lm_{idx}_z",
                f"lm_{idx}_visibility",
                f"lm_{idx}_presence",
            ]
        )
    return fieldnames


def _csv_row(
    frame: int,
    timestamp_ms: int,
    landmarks: list[Any] | None,
    total_landmarks: int = TOTAL_POSE_LANDMARKS,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame": frame,
        "timestamp_ms": timestamp_ms,
        "pose_detected": landmarks is not None,
    }
    for idx in range(total_landmarks):
        landmark = landmarks[idx] if landmarks is not None and idx < len(landmarks) else None
        if landmark is None:
            row[f"lm_{idx}_x"] = None
            row[f"lm_{idx}_y"] = None
            row[f"lm_{idx}_z"] = None
            row[f"lm_{idx}_visibility"] = None
            row[f"lm_{idx}_presence"] = None
            continue

        lm = _landmark_to_dict(landmark)
        row[f"lm_{idx}_x"] = lm["x"]
        row[f"lm_{idx}_y"] = lm["y"]
        row[f"lm_{idx}_z"] = lm["z"]
        row[f"lm_{idx}_visibility"] = lm["visibility"]
        row[f"lm_{idx}_presence"] = lm["presence"]
    return row


def _json_payload(frame: int, timestamp_ms: int, landmarks: list[Any]) -> dict[str, Any]:
    return {
        "frame": frame,
        "timestamp_ms": timestamp_ms,
        "landmarks": [_landmark_to_dict(lm) for lm in landmarks],
    }


def _iter_movement_videos(movements_dir: Path) -> list[Path]:
    if not movements_dir.exists():
        return []
    video_paths: list[Path] = []
    for child in sorted(movements_dir.iterdir()):
        if not child.is_dir():
            continue
        video_path = child / "video.mp4"
        if video_path.exists():
            video_paths.append(video_path)
    return video_paths


def _process_video(
    video_path: Path,
    csv_path: Path,
    json_path: Path,
    landmarker: Any,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_time_ms = int(round(1000.0 / fps))
    frame_index = 0
    timestamp_ms = 0
    first_detected_payload: dict[str, Any] | None = None

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_csv_fieldnames())
        writer.writeheader()

        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None

                writer.writerow(
                    _csv_row(
                        frame=frame_index,
                        timestamp_ms=timestamp_ms,
                        landmarks=landmarks,
                    )
                )

                if first_detected_payload is None and landmarks is not None:
                    first_detected_payload = _json_payload(
                        frame=frame_index,
                        timestamp_ms=timestamp_ms,
                        landmarks=landmarks,
                    )

                frame_index += 1
                timestamp_ms += frame_time_ms
        finally:
            capture.release()

    payload = (
        first_detected_payload
        if first_detected_payload is not None
        else {"frame": 0, "timestamp_ms": 0, "landmarks": []}
    )
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)


class MediaPipeMovementLandmarksGenerator:
    def __init__(self, movements_dir: Path, model_path: Path) -> None:
        self._movements_dir = movements_dir
        self._model_path = model_path

    def generate(self) -> LandmarksGenerationResult:
        video_paths = _iter_movement_videos(self._movements_dir)
        if not video_paths:
            return LandmarksGenerationResult(
                generated_movements=[],
                errors=[f"No video.mp4 files found under: {self._movements_dir}"],
            )

        ensure_model_exists(model_path=self._model_path)

        base_options = mp.tasks.BaseOptions
        pose_landmarker = mp.tasks.vision.PoseLandmarker
        pose_landmarker_options = mp.tasks.vision.PoseLandmarkerOptions
        running_mode = mp.tasks.vision.RunningMode

        options = pose_landmarker_options(
            base_options=base_options(model_asset_path=str(self._model_path)),
            running_mode=running_mode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        generated_movements: list[str] = []
        errors: list[str] = []

        for video_path in video_paths:
            movement_dir = video_path.parent
            movement_name = movement_dir.name
            csv_path = movement_dir / "landmarks.csv"
            json_path = movement_dir / "landmarks.json"
            try:
                # Create a fresh landmarker per video so timestamps can restart at 0.
                with pose_landmarker.create_from_options(options) as landmarker:
                    _process_video(
                        video_path=video_path,
                        csv_path=csv_path,
                        json_path=json_path,
                        landmarker=landmarker,
                    )
                generated_movements.append(movement_name)
            except Exception as exc:
                errors.append(f"{movement_name}: {exc}")

        return LandmarksGenerationResult(
            generated_movements=generated_movements,
            errors=errors,
        )
