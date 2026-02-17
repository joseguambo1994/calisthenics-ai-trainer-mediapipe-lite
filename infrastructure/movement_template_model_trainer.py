from pathlib import Path

from domain.models import MovementModelTrainingResult
from infrastructure.movement_ml_classifier import MovementKNNClassifier


def _normalize_movement_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-").replace(" ", "-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    if normalized == "wing-360":
        return "swing-360"
    return normalized


def _resolve_template_csvs(
    movements_root: Path,
) -> dict[str, list[Path]]:
    if not movements_root.exists():
        return {}

    baselines: dict[str, list[Path]] = {}
    for child in sorted(movements_root.iterdir()):
        if not child.is_dir():
            continue
        movement_name = _normalize_movement_name(child.name)
        csv_paths = sorted(child.rglob("landmarks.csv"))
        if csv_paths:
            baselines[movement_name] = csv_paths
    return baselines


class MovementTemplateModelTrainer:
    def __init__(
        self,
        movements_dir: Path,
        model_path: Path,
        default_k: int = 7,
    ) -> None:
        self._movements_dir = movements_dir
        self._model_path = model_path
        self._default_k = max(1, int(default_k))

    def train(
        self,
        k: int | None = None,
    ) -> MovementModelTrainingResult:
        train_k = self._default_k if k is None else max(1, int(k))
        baseline_csvs = _resolve_template_csvs(self._movements_dir)
        if not baseline_csvs:
            message = f"No landmarks.csv templates found under {self._movements_dir}"
            return MovementModelTrainingResult(
                model_path=str(self._model_path),
                movements_trained=[],
                template_files=0,
                errors=[message],
            )

        classifier = MovementKNNClassifier(baseline_csvs=baseline_csvs, k=train_k)
        if not classifier.has_model:
            return MovementModelTrainingResult(
                model_path=str(self._model_path),
                movements_trained=[],
                template_files=0,
                errors=["Training produced an empty model. Check landmarks.csv contents."],
            )

        try:
            classifier.save_model(self._model_path)
        except Exception as exc:
            return MovementModelTrainingResult(
                model_path=str(self._model_path),
                movements_trained=[],
                template_files=0,
                errors=[str(exc)],
            )

        return MovementModelTrainingResult(
            model_path=str(self._model_path),
            movements_trained=sorted(baseline_csvs.keys()),
            template_files=sum(len(paths) for paths in baseline_csvs.values()),
            errors=[],
        )
