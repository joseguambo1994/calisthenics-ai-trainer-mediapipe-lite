from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging
from pydantic import BaseModel, Field

from api.dependencies import (
    get_evaluate_movement_model_use_case,
    get_generate_landmarks_use_case,
    get_train_movement_model_use_case,
    get_use_case,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="Calisthenics AI Trainer API", version="1.0.0")


class ProcessVideoRequest(BaseModel):
    file_id: str = Field(min_length=1, description="Telegram file_id of the input video")


class ProcessVideoResponse(BaseModel):
    r2_url: str
    video_signed_url: str
    movement_name: str
    technique_feedback: list[str]
    technique_similarity_percent: float
    template_landmarks_source: str | None = Field(
        default=None,
        description="Template landmarks CSV used as reference for the predicted movement.",
    )
    template_landmarks: list[dict[int, tuple[float, float]]] = Field(
        default_factory=list,
        description="Template landmark frames used to draw the ideal execution pose sequence.",
    )


class GenerateLandmarksSuccessResponse(BaseModel):
    generated_movements: list[str]


class GenerateLandmarksErrorResponse(BaseModel):
    errors: list[str]


class TrainMovementModelRequest(BaseModel):
    k: int | None = Field(
        default=None,
        ge=1,
        description="Optional k value for the KNN template model.",
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {"k": 7},
                {},
            ]
        }
    }


class TrainMovementModelSuccessResponse(BaseModel):
    model_path: str
    movements_trained: list[str]
    template_files: int


class TrainMovementModelErrorResponse(BaseModel):
    errors: list[str]


class EvaluateMovementModelRequest(BaseModel):
    regenerate_landmarks: bool = Field(
        default=False,
        description="Regenerate landmarks.csv in the evaluation dataset before scoring.",
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {"regenerate_landmarks": True},
                {},
            ]
        }
    }


class MovementModelSkippedSampleResponse(BaseModel):
    sample_path: str
    true_label: str
    reason: str


class MovementModelEvaluationSampleResponse(BaseModel):
    sample_path: str
    true_label: str
    predicted_label: str
    similarity_percent: float
    valid_rows: int


class MovementClassMetricsResponse(BaseModel):
    label: str
    precision: float
    recall: float
    f1_score: float
    support: int


class EvaluateMovementModelSuccessResponse(BaseModel):
    model_path: str
    evaluation_dir: str
    labels: list[str]
    model_labels: list[str]
    evaluated_samples: int
    skipped_samples: list[MovementModelSkippedSampleResponse]
    confusion_matrix: list[list[int]]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1_score: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1_score: float
    per_class_metrics: list[MovementClassMetricsResponse]
    samples: list[MovementModelEvaluationSampleResponse]


class EvaluateMovementModelErrorResponse(BaseModel):
    errors: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/videos/process", response_model=ProcessVideoResponse)
def process_video(payload: ProcessVideoRequest) -> ProcessVideoResponse:
    try:
        use_case = get_use_case()
        result = use_case.execute(file_id=payload.file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    return ProcessVideoResponse(
        r2_url=result.object_url,
        video_signed_url=result.video_signed_url,
        movement_name=result.movement_name,
        technique_feedback=result.technique_feedback,
        technique_similarity_percent=result.technique_similarity_percent,
        template_landmarks_source=result.template_landmarks_source,
        template_landmarks=result.template_landmarks,
    )


@app.post(
    "/landmarks/generate",
    response_model=GenerateLandmarksSuccessResponse,
    responses={
        500: {
            "model": GenerateLandmarksErrorResponse,
            "description": "Landmarks generation failed.",
        }
    },
)
def generate_landmarks() -> GenerateLandmarksSuccessResponse | JSONResponse:
    try:
        use_case = get_generate_landmarks_use_case()
        result = use_case.execute()
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=GenerateLandmarksErrorResponse(errors=[str(exc)]).model_dump(),
        )

    if result.errors:
        return JSONResponse(
            status_code=500,
            content=GenerateLandmarksErrorResponse(errors=result.errors).model_dump(),
        )

    return GenerateLandmarksSuccessResponse(generated_movements=result.generated_movements)


@app.post(
    "/movement-model/train",
    response_model=TrainMovementModelSuccessResponse,
    responses={
        500: {
            "model": TrainMovementModelErrorResponse,
            "description": "Movement model training failed.",
        }
    },
)
def train_movement_model(
    payload: TrainMovementModelRequest,
) -> TrainMovementModelSuccessResponse | JSONResponse:
    try:
        use_case = get_train_movement_model_use_case()
        result = use_case.execute(
            k=payload.k,
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=TrainMovementModelErrorResponse(errors=[str(exc)]).model_dump(),
        )

    if result.errors:
        return JSONResponse(
            status_code=500,
            content=TrainMovementModelErrorResponse(errors=result.errors).model_dump(),
        )

    return TrainMovementModelSuccessResponse(
        model_path=result.model_path,
        movements_trained=result.movements_trained,
        template_files=result.template_files,
    )


@app.post(
    "/movement-model/evaluate",
    response_model=EvaluateMovementModelSuccessResponse,
    responses={
        500: {
            "model": EvaluateMovementModelErrorResponse,
            "description": "Movement model evaluation failed.",
        }
    },
)
def evaluate_movement_model(
    payload: EvaluateMovementModelRequest,
) -> EvaluateMovementModelSuccessResponse | JSONResponse:
    try:
        use_case = get_evaluate_movement_model_use_case()
        result = use_case.execute(
            regenerate_landmarks=payload.regenerate_landmarks,
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=EvaluateMovementModelErrorResponse(errors=[str(exc)]).model_dump(),
        )

    if result.errors:
        return JSONResponse(
            status_code=500,
            content=EvaluateMovementModelErrorResponse(errors=result.errors).model_dump(),
        )

    return EvaluateMovementModelSuccessResponse(
        model_path=result.model_path,
        evaluation_dir=result.evaluation_dir,
        labels=result.labels,
        model_labels=result.model_labels,
        evaluated_samples=result.evaluated_samples,
        skipped_samples=[
            MovementModelSkippedSampleResponse(
                sample_path=sample.sample_path,
                true_label=sample.true_label,
                reason=sample.reason,
            )
            for sample in result.skipped_samples
        ],
        confusion_matrix=result.confusion_matrix,
        accuracy=result.accuracy,
        macro_precision=result.macro_precision,
        macro_recall=result.macro_recall,
        macro_f1_score=result.macro_f1_score,
        weighted_precision=result.weighted_precision,
        weighted_recall=result.weighted_recall,
        weighted_f1_score=result.weighted_f1_score,
        per_class_metrics=[
            MovementClassMetricsResponse(
                label=metric.label,
                precision=metric.precision,
                recall=metric.recall,
                f1_score=metric.f1_score,
                support=metric.support,
            )
            for metric in result.per_class_metrics
        ],
        samples=[
            MovementModelEvaluationSampleResponse(
                sample_path=sample.sample_path,
                true_label=sample.true_label,
                predicted_label=sample.predicted_label,
                similarity_percent=sample.similarity_percent,
                valid_rows=sample.valid_rows,
            )
            for sample in result.samples
        ],
    )
