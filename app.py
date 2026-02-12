from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_use_case

app = FastAPI(title="Calisthenics AI Trainer API", version="1.0.0")


class ProcessVideoRequest(BaseModel):
    file_id: str = Field(min_length=1, description="Telegram file_id of the input video")


class ProcessVideoResponse(BaseModel):
    r2_url: str
    video_signed_url: str


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
    )
