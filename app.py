from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging
from pydantic import BaseModel, Field

from api.dependencies import get_generate_landmarks_use_case, get_use_case

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


class GenerateLandmarksSuccessResponse(BaseModel):
    generated_movements: list[str]


class GenerateLandmarksErrorResponse(BaseModel):
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
