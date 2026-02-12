import os
from functools import lru_cache
from pathlib import Path

from application.use_cases.process_telegram_video import ProcessTelegramVideoUseCase
from infrastructure.cloudflare_r2_storage import CloudflareR2StorageGateway
from infrastructure.mediapipe_pose_processor import MediaPipePoseVideoProcessor
from infrastructure.telegram_bot_gateway import TelegramBotGateway


@lru_cache(maxsize=1)
def get_use_case() -> ProcessTelegramVideoUseCase:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("Missing environment variable TELEGRAM_BOT_TOKEN")

    r2_account_id = os.getenv("R2_ACCOUNT_ID")
    r2_access_key_id = os.getenv("R2_ACCESS_KEY_ID")
    r2_secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
    r2_bucket_name = os.getenv("R2_BUCKET_NAME")
    if not all([r2_account_id, r2_access_key_id, r2_secret_access_key, r2_bucket_name]):
        raise RuntimeError(
            "Missing Cloudflare R2 environment variables. "
            "Expected: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME"
        )

    r2_endpoint_url = os.getenv(
        "R2_ENDPOINT_URL",
        f"https://{r2_account_id}.r2.cloudflarestorage.com",
    )
    r2_public_base_url = os.getenv("R2_PUBLIC_BASE_URL")
    signed_url_expiration_seconds = int(os.getenv("R2_SIGNED_URL_EXPIRATION_SECONDS", "3600"))

    model_path = Path(os.getenv("MODEL_PATH", "pose_landmarker_lite.task"))
    workspace = Path(os.getenv("WORKSPACE_DIR", "workdir"))

    gateway = TelegramBotGateway(bot_token=bot_token)
    processor = MediaPipePoseVideoProcessor(model_path=model_path)
    storage = CloudflareR2StorageGateway(
        endpoint_url=r2_endpoint_url,
        access_key_id=r2_access_key_id,
        secret_access_key=r2_secret_access_key,
        bucket_name=r2_bucket_name,
        public_base_url=r2_public_base_url,
        signed_url_expiration_seconds=signed_url_expiration_seconds,
    )

    return ProcessTelegramVideoUseCase(
        gateway=gateway,
        processor=processor,
        storage=storage,
        workspace_dir=workspace,
    )
