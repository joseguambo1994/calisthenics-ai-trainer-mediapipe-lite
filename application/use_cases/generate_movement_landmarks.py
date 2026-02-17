from domain.models import LandmarksGenerationResult
from domain.ports import MovementLandmarksGenerator


class GenerateMovementLandmarksUseCase:
    def __init__(self, generator: MovementLandmarksGenerator) -> None:
        self._generator = generator

    def execute(self) -> LandmarksGenerationResult:
        return self._generator.generate()
