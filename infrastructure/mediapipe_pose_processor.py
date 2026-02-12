from pathlib import Path

import cv2
import mediapipe as mp

from domain.models import ProcessedVideo
from infrastructure.model_downloader import ensure_model_exists
from infrastructure.movement_detector_and_feedback import (
    FrameFeatures,
    build_feedback,
    compute_frame_features,
    detect_movement,
)

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (27, 31), (28, 32),
]


def draw_pose(frame_bgr, pose_landmarks, point_radius: int = 2, line_thickness: int = 2) -> None:
    height, width = frame_bgr.shape[:2]

    for start_idx, end_idx in POSE_CONNECTIONS:
        start_landmark = pose_landmarks[start_idx]
        end_landmark = pose_landmarks[end_idx]

        start_x, start_y = int(start_landmark.x * width), int(start_landmark.y * height)
        end_x, end_y = int(end_landmark.x * width), int(end_landmark.y * height)

        cv2.line(frame_bgr, (start_x, start_y), (end_x, end_y), (0, 255, 0), line_thickness)

    for landmark in pose_landmarks:
        x, y = int(landmark.x * width), int(landmark.y * height)
        cv2.circle(frame_bgr, (x, y), point_radius, (0, 0, 255), -1)


class MediaPipePoseVideoProcessor:
    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        ensure_model_exists(model_path=self._model_path)

    def process(self, input_video: Path, output_video: Path) -> ProcessedVideo:
        if not input_video.exists():
            raise FileNotFoundError(f"Missing: {input_video}")

        if not self._model_path.exists():
            raise FileNotFoundError(f"Missing: {self._model_path}")

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

        capture = cv2.VideoCapture(str(input_video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {input_video}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open writer: {output_video}")

        timestamp_ms = 0
        frame_time_ms = int(round(1000.0 / fps))
        frames = 0
        features: list[FrameFeatures] = []
        previous_shoulder_angle: float | None = None

        try:
            with pose_landmarker.create_from_options(options) as landmarker:
                while True:
                    ok, frame_bgr = capture.read()
                    if not ok:
                        break

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[0]
                        draw_pose(frame_bgr, landmarks)
                        frame_features, previous_shoulder_angle = compute_frame_features(
                            landmarks=landmarks,
                            previous_shoulder_angle=previous_shoulder_angle,
                        )
                        features.append(frame_features)

                    writer.write(frame_bgr)
                    frames += 1
                    timestamp_ms += frame_time_ms
        finally:
            capture.release()
            writer.release()

        movement_name = detect_movement(features)
        technique_feedback = build_feedback(movement_name, features)
        return ProcessedVideo(
            output_path=str(output_video),
            frames=frames,
            fps=fps,
            movement_name=movement_name,
            technique_feedback=technique_feedback,
        )
