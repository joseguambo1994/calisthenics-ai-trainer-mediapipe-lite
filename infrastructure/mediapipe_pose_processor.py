from pathlib import Path
import os
from typing import Any

import cv2
import mediapipe as mp

from domain.models import ProcessedVideo
from infrastructure.landmark_baseline_evaluator import LandmarkBaselineEvaluator
from infrastructure.model_downloader import ensure_model_exists
from infrastructure.movement_ml_classifier import MovementKNNClassifier
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
    draw_pose_with_color(
        frame_bgr=frame_bgr,
        pose_landmarks=pose_landmarks,
        line_color=(0, 255, 0),
        point_color=(0, 0, 255),
        point_radius=point_radius,
        line_thickness=line_thickness,
    )


def draw_pose_with_color(
    frame_bgr,
    pose_landmarks,
    line_color: tuple[int, int, int],
    point_color: tuple[int, int, int],
    point_radius: int = 2,
    line_thickness: int = 2,
) -> None:
    height, width = frame_bgr.shape[:2]

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx >= len(pose_landmarks) or end_idx >= len(pose_landmarks):
            continue
        start_landmark = pose_landmarks[start_idx]
        end_landmark = pose_landmarks[end_idx]
        if start_landmark is None or end_landmark is None:
            continue

        start_x, start_y = int(start_landmark.x * width), int(start_landmark.y * height)
        end_x, end_y = int(end_landmark.x * width), int(end_landmark.y * height)

        cv2.line(frame_bgr, (start_x, start_y), (end_x, end_y), line_color, line_thickness)

    for landmark in pose_landmarks:
        if landmark is None:
            continue
        x, y = int(landmark.x * width), int(landmark.y * height)
        cv2.circle(frame_bgr, (x, y), point_radius, point_color, -1)


def _extract_points(landmarks: list[Any]) -> dict[int, tuple[float, float]]:
    return {idx: (float(lm.x), float(lm.y)) for idx, lm in enumerate(landmarks)}


def _build_landmarks_for_drawing(points: dict[int, tuple[float, float]], total_landmarks: int = 33) -> list[Any]:
    class _Point:
        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y

    result: list[Any] = [None] * total_landmarks
    for idx, (x, y) in points.items():
        if 0 <= idx < total_landmarks:
            result[idx] = _Point(x=x, y=y)
    return result


def _resolve_baseline_csv_path() -> Path | None:
    env_path = os.getenv("BASELINE_LANDMARKS_CSV_PATH", "").strip()
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate

    search_roots = [
        Path("workdir2") / "landmarks",
        Path("videoslocalesswing360") / "landmarks",
    ]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(root.rglob("*.landmarks.csv"))

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _normalize_movement_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-").replace(" ", "-")
    normalized = normalized.replace("--", "-")
    if normalized == "wing-360":
        return "swing-360"
    return normalized


def _resolve_movement_baseline_csv_paths() -> dict[str, list[Path]]:
    movements_root = Path("movements")
    if not movements_root.exists():
        return {}

    env_names = os.getenv("MOVEMENT_BASELINES", "").strip()
    if env_names:
        requested = [_normalize_movement_name(item) for item in env_names.split(",") if item.strip()]
    else:
        # Defaults requested for calisthenics movements in this project.
        requested = ["swing-360", "strict-muscle-up", "olympic-muscle-up", "handstand"]

    baselines: dict[str, list[Path]] = {}
    for movement_name in requested:
        movement_dir = movements_root / movement_name
        if not movement_dir.exists() or not movement_dir.is_dir():
            continue
        csv_paths = sorted(movement_dir.rglob("landmarks.csv"))
        if csv_paths:
            baselines[movement_name] = csv_paths

    # Fallback to any movement baseline found on disk if no default/requested baseline exists.
    if baselines:
        return baselines

    for child in movements_root.iterdir():
        if not child.is_dir():
            continue
        csv_paths = sorted(child.rglob("landmarks.csv"))
        if csv_paths:
            baselines[_normalize_movement_name(child.name)] = csv_paths
    return baselines


def _resolve_movement_model_path() -> Path:
    model_path = os.getenv("MOVEMENT_MODEL_PATH", "").strip()
    if model_path:
        return Path(model_path)
    return Path("models") / "movement_template_model.npz"


def _resolve_template_landmarks_max_frames() -> int | None:
    raw = os.getenv("TEMPLATE_LANDMARKS_MAX_FRAMES", "180").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 180
    if value <= 0:
        return None
    return value


def _feedback_label_for_movement(movement_name: str) -> str:
    movement_name = _normalize_movement_name(movement_name)
    if movement_name in {"swing-360", "swing360"}:
        return "swing360"
    if movement_name in {"strict-muscle-up", "olympic-muscle-up", "muscleup"}:
        return "muscleup"
    if movement_name == "handstand":
        return "handstand"
    return "unknown"


class MediaPipePoseVideoProcessor:
    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._deviation_red_threshold = float(os.getenv("BASELINE_DEVIATION_RED_THRESHOLD", "0.18"))
        self._max_deviation_for_low_score = float(os.getenv("BASELINE_LOW_SCORE_DEVIATION", "0.25"))
        self._movement_ml_k = int(os.getenv("MOVEMENT_ML_K", "7"))
        self._movement_model_path = _resolve_movement_model_path()
        self._template_landmarks_max_frames = _resolve_template_landmarks_max_frames()
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
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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
        frame_deviation: float | None = None
        baseline_csv_path = _resolve_baseline_csv_path()
        baseline_evaluator = LandmarkBaselineEvaluator([], self._max_deviation_for_low_score)
        if baseline_csv_path is not None:
            baseline_evaluator = LandmarkBaselineEvaluator.from_csv(
                baseline_csv_path,
                max_deviation_for_low_score=self._max_deviation_for_low_score,
            )
        movement_baselines = _resolve_movement_baseline_csv_paths()
        movement_evaluators = {
            movement_name: LandmarkBaselineEvaluator.from_csv(
                paths[0],
                max_deviation_for_low_score=self._max_deviation_for_low_score,
            )
            for movement_name, paths in movement_baselines.items()
            if paths
        }
        movement_template_csvs = {
            movement_name: paths[0]
            for movement_name, paths in movement_baselines.items()
            if paths
        }
        try:
            movement_classifier = MovementKNNClassifier.load_model(self._movement_model_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load trained movement model. "
                "Run /movement-model/train before processing videos."
            ) from exc

        try:
            with pose_landmarker.create_from_options(options) as landmarker:
                while True:
                    ok, frame_bgr = capture.read()
                    if not ok:
                        break

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    frame_deviation = None

                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[0]
                        current_points = _extract_points(landmarks)
                        frame_deviation = baseline_evaluator.add_observation(
                            frame_index=frames,
                            current_total_frames=max(1, total_frames),
                            current_points=current_points,
                        )
                        movement_classifier.add_observation(current_points)
                        current_line_color = (0, 0, 255) if (
                            frame_deviation is not None and frame_deviation > self._deviation_red_threshold
                        ) else (0, 255, 0)
                        draw_pose_with_color(
                            frame_bgr=frame_bgr,
                            pose_landmarks=landmarks,
                            line_color=current_line_color,
                            point_color=(0, 0, 255),
                        )
                        if baseline_evaluator.has_baseline:
                            baseline_points = baseline_evaluator.baseline_points_for_frame(
                                frame_index=frames,
                                current_total_frames=max(1, total_frames),
                            )
                            if baseline_points:
                                baseline_landmarks = _build_landmarks_for_drawing(baseline_points)
                                draw_pose_with_color(
                                    frame_bgr=frame_bgr,
                                    pose_landmarks=baseline_landmarks,
                                    line_color=(255, 255, 0),
                                    point_color=(255, 255, 0),
                                    point_radius=2,
                                    line_thickness=1,
                                )
                        frame_features, previous_shoulder_angle = compute_frame_features(
                            landmarks=landmarks,
                            previous_shoulder_angle=previous_shoulder_angle,
                        )
                        features.append(frame_features)

                    similarity_live = baseline_evaluator.similarity_percent()
                    deviation_text = "n/a" if frame_deviation is None else f"{frame_deviation:.3f}"
                    cv2.putText(
                        frame_bgr,
                        f"Similitud: {similarity_live:.1f}%",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame_bgr,
                        f"Deviation: {deviation_text}",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    writer.write(frame_bgr)
                    frames += 1
                    timestamp_ms += frame_time_ms
        finally:
            capture.release()
            writer.release()

        movement_name = detect_movement(features)
        ml_prediction = movement_classifier.predict()
        if ml_prediction.movement_name != "unknown":
            movement_name = ml_prediction.movement_name
        technique_similarity_percent = ml_prediction.similarity_percent
        technique_feedback = build_feedback(_feedback_label_for_movement(movement_name), features)
        selected_template_source: str | None = None
        selected_template_landmarks: list[dict[int, tuple[float, float]]] = []
        normalized_movement = _normalize_movement_name(movement_name)
        selected_evaluator = movement_evaluators.get(normalized_movement)
        selected_csv = movement_template_csvs.get(normalized_movement)
        if selected_evaluator is not None and selected_evaluator.has_baseline:
            selected_template_landmarks = selected_evaluator.export_baseline_frames(
                max_frames=self._template_landmarks_max_frames,
            )
        if selected_csv is not None:
            selected_template_source = str(selected_csv)
        return ProcessedVideo(
            output_path=str(output_video),
            frames=frames,
            fps=fps,
            movement_name=movement_name,
            technique_feedback=technique_feedback,
            technique_similarity_percent=technique_similarity_percent,
            template_landmarks_source=selected_template_source,
            template_landmarks=selected_template_landmarks,
        )
