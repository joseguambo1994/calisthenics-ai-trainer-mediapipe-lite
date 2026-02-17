from domain.models import MovementModelTrainingResult
from domain.ports import MovementTemplateModelTrainer


class TrainMovementTemplateModelUseCase:
    def __init__(self, trainer: MovementTemplateModelTrainer) -> None:
        self._trainer = trainer

    def execute(
        self,
        k: int | None = None,
    ) -> MovementModelTrainingResult:
        return self._trainer.train(k=k)
