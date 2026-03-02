from __future__ import annotations

import csv
from pathlib import Path

from domain.models import (
    MovementClassMetrics,
    MovementModelEvaluationResult,
    MovementModelEvaluationSample,
    MovementModelSkippedSample,
)
from infrastructure.movement_ml_classifier import KEY_LANDMARKS, MovementKNNClassifier


def _normalize_movement_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-").replace(" ", "-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    if normalized == "wing-360":
        return "swing-360"
    return normalized


class MovementModelMetricsEvaluator:
    def __init__(self, evaluation_dir: Path, model_path: Path) -> None:
        self._evaluation_dir = evaluation_dir
        self._model_path = model_path

    def evaluate(self) -> MovementModelEvaluationResult:
        if not self._evaluation_dir.exists():
            return self._empty_result(errors=[f"Evaluation directory not found: {self._evaluation_dir}"])

        if not self._model_path.exists():
            return self._empty_result(errors=[f"Movement model not found: {self._model_path}"])

        classifier = MovementKNNClassifier.load_model(self._model_path)
        model_labels = sorted(classifier.labels)
        csv_paths = sorted(self._evaluation_dir.rglob("landmarks.csv"))
        if not csv_paths:
            return self._empty_result(model_labels=model_labels, errors=[f"No landmarks.csv found under {self._evaluation_dir}"])

        samples: list[MovementModelEvaluationSample] = []
        skipped_samples: list[MovementModelSkippedSample] = []
        for csv_path in csv_paths:
            relative_csv = csv_path.relative_to(self._evaluation_dir)
            if len(relative_csv.parts) < 2:
                skipped_samples.append(
                    MovementModelSkippedSample(
                        sample_path=relative_csv.as_posix(),
                        true_label="unknown",
                        reason="Expected movement/angle/landmarks.csv structure.",
                    )
                )
                continue

            true_label = _normalize_movement_name(relative_csv.parts[0])
            sample_path = relative_csv.parent.as_posix()
            if true_label not in model_labels:
                skipped_samples.append(
                    MovementModelSkippedSample(
                        sample_path=sample_path,
                        true_label=true_label,
                        reason="True label is not present in the trained model.",
                    )
                )
                continue

            prediction, valid_rows = self._predict_csv(csv_path)
            if valid_rows == 0:
                skipped_samples.append(
                    MovementModelSkippedSample(
                        sample_path=sample_path,
                        true_label=true_label,
                        reason="No valid detected pose rows found in landmarks.csv.",
                    )
                )
                continue

            samples.append(
                MovementModelEvaluationSample(
                    sample_path=sample_path,
                    true_label=true_label,
                    predicted_label=_normalize_movement_name(prediction.movement_name),
                    similarity_percent=float(prediction.similarity_percent),
                    valid_rows=valid_rows,
                )
            )

        if not samples:
            return self._empty_result(
                model_labels=model_labels,
                skipped_samples=skipped_samples,
                errors=["No evaluation samples could be scored against the trained model."],
            )

        labels = sorted({sample.true_label for sample in samples} | {sample.predicted_label for sample in samples})
        label_index = {label: idx for idx, label in enumerate(labels)}
        confusion_matrix = [[0 for _ in labels] for _ in labels]
        for sample in samples:
            confusion_matrix[label_index[sample.true_label]][label_index[sample.predicted_label]] += 1

        total_samples = len(samples)
        correct_predictions = sum(
            confusion_matrix[idx][idx]
            for idx in range(len(labels))
        )
        accuracy = correct_predictions / total_samples

        per_class_metrics: list[MovementClassMetrics] = []
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1_score = 0.0
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1_score = 0.0

        for idx, label in enumerate(labels):
            tp = confusion_matrix[idx][idx]
            fp = sum(confusion_matrix[row_idx][idx] for row_idx in range(len(labels)) if row_idx != idx)
            fn = sum(confusion_matrix[idx][col_idx] for col_idx in range(len(labels)) if col_idx != idx)
            support = sum(confusion_matrix[idx])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                (2.0 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_class_metrics.append(
                MovementClassMetrics(
                    label=label,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    support=support,
                )
            )
            macro_precision += precision
            macro_recall += recall
            macro_f1_score += f1_score
            weighted_precision += precision * support
            weighted_recall += recall * support
            weighted_f1_score += f1_score * support

        class_count = len(per_class_metrics)
        macro_precision /= class_count
        macro_recall /= class_count
        macro_f1_score /= class_count
        weighted_precision /= total_samples
        weighted_recall /= total_samples
        weighted_f1_score /= total_samples

        return MovementModelEvaluationResult(
            model_path=str(self._model_path),
            evaluation_dir=str(self._evaluation_dir),
            labels=labels,
            model_labels=model_labels,
            evaluated_samples=total_samples,
            skipped_samples=skipped_samples,
            confusion_matrix=confusion_matrix,
            accuracy=accuracy,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1_score=macro_f1_score,
            weighted_precision=weighted_precision,
            weighted_recall=weighted_recall,
            weighted_f1_score=weighted_f1_score,
            per_class_metrics=per_class_metrics,
            samples=samples,
            errors=[],
        )

    def _predict_csv(self, csv_path: Path) -> tuple[object, int]:
        classifier = MovementKNNClassifier.load_model(self._model_path)
        valid_rows = 0
        with csv_path.open("r", encoding="utf-8", newline="") as file_handle:
            reader = csv.DictReader(file_handle)
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

                if not points:
                    continue

                classifier.add_observation(points)
                valid_rows += 1

        return classifier.predict(), valid_rows

    def _empty_result(
        self,
        model_labels: list[str] | None = None,
        skipped_samples: list[MovementModelSkippedSample] | None = None,
        errors: list[str] | None = None,
    ) -> MovementModelEvaluationResult:
        return MovementModelEvaluationResult(
            model_path=str(self._model_path),
            evaluation_dir=str(self._evaluation_dir),
            labels=[],
            model_labels=model_labels or [],
            evaluated_samples=0,
            skipped_samples=skipped_samples or [],
            confusion_matrix=[],
            accuracy=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1_score=0.0,
            weighted_precision=0.0,
            weighted_recall=0.0,
            weighted_f1_score=0.0,
            per_class_metrics=[],
            samples=[],
            errors=errors or [],
        )
