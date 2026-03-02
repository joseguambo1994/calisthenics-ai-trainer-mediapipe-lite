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


@dataclass(frozen=True)
class MovementModelEvaluationSample:
    sample_path: str
    true_label: str
    predicted_label: str
    similarity_percent: float
    valid_rows: int


@dataclass(frozen=True)
class MovementModelSkippedSample:
    sample_path: str
    true_label: str
    reason: str


@dataclass(frozen=True)
class MovementClassMetrics:
    label: str
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass(frozen=True)
class MovementModelEvaluationResult:
    model_path: str
    evaluation_dir: str
    labels: list[str]
    model_labels: list[str]
    evaluated_samples: int
    skipped_samples: list[MovementModelSkippedSample]
    confusion_matrix: list[list[int]]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1_score: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1_score: float
    per_class_metrics: list[MovementClassMetrics]
    samples: list[MovementModelEvaluationSample]
    errors: list[str]
