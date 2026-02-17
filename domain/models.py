from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessedVideo:
    output_path: str
    frames: int
    fps: float
    movement_name: str
    technique_feedback: list[str]
    technique_similarity_percent: float
    template_landmarks_source: str | None
    template_landmarks: list[dict[int, tuple[float, float]]]


@dataclass(frozen=True)
class StoredObject:
    object_key: str
    object_url: str
    signed_url: str


@dataclass(frozen=True)
class ProcessedVideoDelivery:
    output_path: str
    frames: int
    fps: float
    movement_name: str
    technique_feedback: list[str]
    technique_similarity_percent: float
    template_landmarks_source: str | None
    template_landmarks: list[dict[int, tuple[float, float]]]
    object_key: str
    object_url: str
    video_signed_url: str


@dataclass(frozen=True)
class LandmarksGenerationResult:
    generated_movements: list[str]
    errors: list[str]


@dataclass(frozen=True)
class MovementModelTrainingResult:
    model_path: str
    movements_trained: list[str]
    template_files: int
    errors: list[str]
