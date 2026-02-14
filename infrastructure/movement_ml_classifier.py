from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KEY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


@dataclass(frozen=True)
class MLMovementPrediction:
    movement_name: str
    similarity_percent: float
    label_scores: dict[str, float]


class MovementKNNClassifier:
    def __init__(self, baseline_csvs: dict[str, Path], k: int = 7) -> None:
        self._k = max(1, int(k))
        self._labels: list[str] = []
        self._x_train = np.zeros((0, len(KEY_LANDMARKS) * 2), dtype=np.float32)
        self._y_train: list[str] = []
        self._score_sums: dict[str, float] = {}
        self._distance_sums: dict[str, float] = {}
        self._distance_counts: dict[str, int] = {}
        self._fit_from_csvs(baseline_csvs)

    @property
    def has_model(self) -> bool:
        return self._x_train.shape[0] > 0

    def add_observation(self, points: dict[int, tuple[float, float]]) -> None:
        vector = self._vectorize_points(points)
        if vector is None or not self.has_model:
            return

        distances = np.linalg.norm(self._x_train - vector, axis=1)
        k = min(self._k, len(distances))
        idxs = np.argpartition(distances, k - 1)[:k]
        nearest_distances = distances[idxs]
        nearest_labels = [self._y_train[i] for i in idxs]

        # Inverse-distance weighted vote over k-nearest examples.
        eps = 1e-6
        for label, distance in zip(nearest_labels, nearest_distances):
            weight = float(1.0 / (distance + eps))
            self._score_sums[label] = self._score_sums.get(label, 0.0) + weight

        for label in self._labels:
            label_mask = np.array([y == label for y in self._y_train], dtype=bool)
            if not np.any(label_mask):
                continue
            min_distance = float(np.min(distances[label_mask]))
            self._distance_sums[label] = self._distance_sums.get(label, 0.0) + min_distance
            self._distance_counts[label] = self._distance_counts.get(label, 0) + 1

    def predict(self) -> MLMovementPrediction:
        if not self.has_model or not self._score_sums:
            return MLMovementPrediction(
                movement_name="unknown",
                similarity_percent=10.0,
                label_scores={},
            )

        total = float(sum(self._score_sums.values())) + 1e-6
        label_scores = {label: (score / total) * 100.0 for label, score in self._score_sums.items()}
        best_label = max(label_scores, key=label_scores.get)

        # Distance-based similarity score calibrated to 10-100 range.
        avg_distance = float(
            self._distance_sums.get(best_label, 1.0)
            / max(1, self._distance_counts.get(best_label, 1))
        )
        similarity = float(np.clip(100.0 * np.exp(-0.5 * avg_distance), 10.0, 100.0))

        return MLMovementPrediction(
            movement_name=best_label,
            similarity_percent=similarity,
            label_scores=label_scores,
        )

    def _fit_from_csvs(self, baseline_csvs: dict[str, Path]) -> None:
        vectors: list[np.ndarray] = []
        labels: list[str] = []

        for movement_name, csv_path in baseline_csvs.items():
            if not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("pose_detected", "")).lower() not in {"true", "1"}:
                        continue
                    points: dict[int, tuple[float, float]] = {}
                    for idx in KEY_LANDMARKS:
                        x = row.get(f"lm_{idx}_x")
                        y = row.get(f"lm_{idx}_y")
                        if x in (None, "", "None") or y in (None, "", "None"):
                            continue
                        points[idx] = (float(x), float(y))
                    vector = self._vectorize_points(points)
                    if vector is None:
                        continue
                    vectors.append(vector)
                    labels.append(movement_name)

        if not vectors:
            return
        self._x_train = np.vstack(vectors).astype(np.float32)
        self._y_train = labels
        self._labels = sorted(set(labels))

    @staticmethod
    def _vectorize_points(points: dict[int, tuple[float, float]]) -> np.ndarray | None:
        ls = points.get(11)
        rs = points.get(12)
        if ls is None or rs is None:
            return None

        shoulder_center = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        shoulder_width = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1]))
        if shoulder_width <= 1e-6:
            return None

        values: list[float] = []
        for idx in KEY_LANDMARKS:
            point = points.get(idx)
            if point is None:
                values.extend([0.0, 0.0])
                continue
            values.extend(
                [
                    float((point[0] - shoulder_center[0]) / shoulder_width),
                    float((point[1] - shoulder_center[1]) / shoulder_width),
                ]
            )
        return np.array(values, dtype=np.float32)
