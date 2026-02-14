from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Compare the major joints used for movement quality.
KEY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
ALL_LANDMARKS = list(range(33))


@dataclass(frozen=True)
class BaselineFrame:
    points: dict[int, tuple[float, float]]


class LandmarkBaselineEvaluator:
    def __init__(self, baseline_frames: list[BaselineFrame], max_deviation_for_low_score: float = 0.25) -> None:
        self._baseline_frames = baseline_frames
        self._max_deviation_for_low_score = max(max_deviation_for_low_score, 1e-6)
        self._frame_deviations: list[float] = []

    @property
    def has_baseline(self) -> bool:
        return len(self._baseline_frames) > 0

    def baseline_points_for_frame(self, frame_index: int, current_total_frames: int) -> dict[int, tuple[float, float]] | None:
        if not self._baseline_frames:
            return None
        mapped_idx = self._map_index(frame_index, current_total_frames)
        return self._baseline_frames[mapped_idx].points

    def add_observation(
        self,
        frame_index: int,
        current_total_frames: int,
        current_points: dict[int, tuple[float, float]],
    ) -> float | None:
        baseline_points = self.baseline_points_for_frame(frame_index, current_total_frames)
        if baseline_points is None:
            return None

        shoulder_width = self._distance_from_points(current_points, 11, 12)
        if shoulder_width <= 1e-6:
            return None

        deviations: list[float] = []
        for idx in KEY_LANDMARKS:
            current = current_points.get(idx)
            baseline = baseline_points.get(idx)
            if current is None or baseline is None:
                continue
            deviations.append(float(np.linalg.norm(np.array(current) - np.array(baseline)) / shoulder_width))

        if not deviations:
            return None

        frame_deviation = float(np.mean(deviations))
        self._frame_deviations.append(frame_deviation)
        return frame_deviation

    def similarity_percent(self) -> float:
        if not self._frame_deviations:
            return 10.0
        mean_deviation = float(np.mean(self._frame_deviations))
        normalized = float(np.clip(mean_deviation / self._max_deviation_for_low_score, 0.0, 1.0))
        # 100% means very close to baseline, 10% means too much deviation.
        return float(np.clip(100.0 - 90.0 * normalized, 10.0, 100.0))

    def _map_index(self, frame_index: int, current_total_frames: int) -> int:
        if len(self._baseline_frames) == 1:
            return 0
        ratio = frame_index / max(1, current_total_frames - 1)
        idx = int(round(ratio * (len(self._baseline_frames) - 1)))
        return int(np.clip(idx, 0, len(self._baseline_frames) - 1))

    @staticmethod
    def _distance_from_points(points: dict[int, tuple[float, float]], idx_a: int, idx_b: int) -> float:
        a = points.get(idx_a)
        b = points.get(idx_b)
        if a is None or b is None:
            return 0.0
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    @classmethod
    def from_csv(cls, path: Path, max_deviation_for_low_score: float = 0.25) -> "LandmarkBaselineEvaluator":
        if not path.exists():
            return cls([], max_deviation_for_low_score=max_deviation_for_low_score)

        frames: list[BaselineFrame] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("pose_detected", "")).lower() not in {"true", "1"}:
                    continue
                points: dict[int, tuple[float, float]] = {}
                for idx in ALL_LANDMARKS:
                    x = row.get(f"lm_{idx}_x")
                    y = row.get(f"lm_{idx}_y")
                    if x in (None, "", "None") or y in (None, "", "None"):
                        continue
                    points[idx] = (float(x), float(y))
                if points:
                    frames.append(BaselineFrame(points=points))
        return cls(frames, max_deviation_for_low_score=max_deviation_for_low_score)
