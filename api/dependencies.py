import os
from functools import lru_cache
from pathlib import Path

from application.use_cases.generate_movement_landmarks import GenerateMovementLandmarksUseCase
from application.use_cases.evaluate_movement_model import EvaluateMovementModelUseCase
from application.use_cases.process_telegram_video import ProcessTelegramVideoUseCase
from application.use_cases.train_movement_template_model import TrainMovementTemplateModelUseCase
from infrastructure.cloudflare_r2_storage import CloudflareR2StorageGateway
from infrastructure.mediapipe_pose_processor import MediaPipePoseVideoProcessor
from infrastructure.movement_landmarks_generator import MediaPipeMovementLandmarksGenerator
from infrastructure.movement_model_evaluator import MovementModelMetricsEvaluator
from infrastructure.movement_template_model_trainer import MovementTemplateModelTrainer
from infrastructure.telegram_bot_gateway import TelegramBotGateway


def _resolve_movements_dir() -> Path:
    env_path = os.getenv("MOVEMENTS_DIR", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path("movements"))
    # Fallback to repo-relative path in case process cwd is different.
    candidates.append(Path(__file__).resolve().parents[1] / "movements")

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return candidates[0]


def _resolve_movements_evaluation_dir() -> Path:
    env_path = os.getenv("MOVEMENTS_EVALUATION_DIR", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path("movements-evaluation"))
    candidates.append(Path(__file__).resolve().parents[1] / "movements-evaluation")

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return candidates[0]


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


@lru_cache(maxsize=1)
def get_generate_landmarks_use_case() -> GenerateMovementLandmarksUseCase:
    movements_dir = _resolve_movements_dir()
    model_path = Path(os.getenv("MODEL_PATH", "pose_landmarker_lite.task"))
    generator = MediaPipeMovementLandmarksGenerator(
        movements_dir=movements_dir,
        model_path=model_path,
    )
    return GenerateMovementLandmarksUseCase(generator=generator)


@lru_cache(maxsize=1)
def get_train_movement_model_use_case() -> TrainMovementTemplateModelUseCase:
    movements_dir = _resolve_movements_dir()
    movement_model_path = Path(os.getenv("MOVEMENT_MODEL_PATH", "models/movement_template_model.npz"))
    movement_k = int(os.getenv("MOVEMENT_ML_K", "7"))
    trainer = MovementTemplateModelTrainer(
        movements_dir=movements_dir,
        model_path=movement_model_path,
        default_k=movement_k,
    )
    return TrainMovementTemplateModelUseCase(trainer=trainer)


@lru_cache(maxsize=1)
def get_evaluate_movement_model_use_case() -> EvaluateMovementModelUseCase:
    evaluation_dir = _resolve_movements_evaluation_dir()
    movement_model_path = Path(os.getenv("MOVEMENT_MODEL_PATH", "models/movement_template_model.npz"))
    pose_model_path = Path(os.getenv("MODEL_PATH", "pose_landmarker_lite.task"))
    evaluator = MovementModelMetricsEvaluator(
        evaluation_dir=evaluation_dir,
        model_path=movement_model_path,
    )
    generator = MediaPipeMovementLandmarksGenerator(
        movements_dir=evaluation_dir,
        model_path=pose_model_path,
    )
    return EvaluateMovementModelUseCase(
        evaluator=evaluator,
        generator=generator,
    )
