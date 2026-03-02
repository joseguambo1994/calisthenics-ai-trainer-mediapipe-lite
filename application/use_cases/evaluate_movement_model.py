from domain.models import MovementModelEvaluationResult
from domain.ports import MovementLandmarksGenerator, MovementModelEvaluator


class EvaluateMovementModelUseCase:
    def __init__(
        self,
        evaluator: MovementModelEvaluator,
        generator: MovementLandmarksGenerator | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._generator = generator

    def execute(
        self,
        regenerate_landmarks: bool = False,
    ) -> MovementModelEvaluationResult:
        generation_errors: list[str] = []
        if regenerate_landmarks:
            if self._generator is None:
                generation_errors.append("Evaluation landmarks generator is not configured.")
            else:
                generation_result = self._generator.generate()
                generation_errors.extend(generation_result.errors)

        result = self._evaluator.evaluate()
        if not generation_errors:
            return result

        return MovementModelEvaluationResult(
            model_path=result.model_path,
            evaluation_dir=result.evaluation_dir,
            labels=result.labels,
            model_labels=result.model_labels,
            evaluated_samples=result.evaluated_samples,
            skipped_samples=result.skipped_samples,
            confusion_matrix=result.confusion_matrix,
            accuracy=result.accuracy,
            macro_precision=result.macro_precision,
            macro_recall=result.macro_recall,
            macro_f1_score=result.macro_f1_score,
            weighted_precision=result.weighted_precision,
            weighted_recall=result.weighted_recall,
            weighted_f1_score=result.weighted_f1_score,
            per_class_metrics=result.per_class_metrics,
            samples=result.samples,
            errors=[*generation_errors, *result.errors],
        )
