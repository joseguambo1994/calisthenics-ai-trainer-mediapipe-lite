import os
from functools import lru_cache
from pathlib import Path

from application.use_cases.process_telegram_video import ProcessTelegramVideoUseCase
from infrastructure.mediapipe_pose_processor import MediaPipePoseVideoProcessor
from infrastructure.telegram_bot_gateway import TelegramBotGateway


@lru_cache(maxsize=1)
def get_use_case() -> ProcessTelegramVideoUseCase:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("Missing environment variable TELEGRAM_BOT_TOKEN")

    model_path = Path(os.getenv("MODEL_PATH", "pose_landmarker_lite.task"))
    workspace = Path(os.getenv("WORKSPACE_DIR", "workdir"))

    gateway = TelegramBotGateway(bot_token=bot_token)
    processor = MediaPipePoseVideoProcessor(model_path=model_path)

    return ProcessTelegramVideoUseCase(
        gateway=gateway,
        processor=processor,
        workspace_dir=workspace,
    )
