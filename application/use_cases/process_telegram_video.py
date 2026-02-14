import shutil
from pathlib import Path
from uuid import uuid4

from domain.models import ProcessedVideoDelivery
from domain.ports import ObjectStorageGateway, PoseVideoProcessor, TelegramVideoGateway


class ProcessTelegramVideoUseCase:
    def __init__(
        self,
        gateway: TelegramVideoGateway,
        processor: PoseVideoProcessor,
        storage: ObjectStorageGateway,
        workspace_dir: Path,
    ) -> None:
        self._gateway = gateway
        self._processor = processor
        self._storage = storage
        self._workspace_dir = workspace_dir

    def execute(self, file_id: str) -> ProcessedVideoDelivery:
        request_id = uuid4().hex
        incoming_dir = self._workspace_dir / "incoming" / request_id
        processed_dir = self._workspace_dir / "processed"
        input_video: Path | None = None
        output_video: Path | None = None

        incoming_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        try:
            input_video = self._gateway.download_video(file_id=file_id, target_dir=incoming_dir)
            output_video = processed_dir / f"{input_video.stem}-processed.mp4"
            processed_video = self._processor.process(input_video=input_video, output_video=output_video)

            object_key = f"processed/{request_id}/{output_video.name}"
            stored_object = self._storage.upload_file(
                source_path=Path(processed_video.output_path),
                object_key=object_key,
                content_type="video/mp4",
            )

            return ProcessedVideoDelivery(
                output_path=processed_video.output_path,
                frames=processed_video.frames,
                fps=processed_video.fps,
                movement_name=processed_video.movement_name,
                technique_feedback=processed_video.technique_feedback,
                technique_similarity_percent=processed_video.technique_similarity_percent,
                object_key=stored_object.object_key,
                object_url=stored_object.object_url,
                video_signed_url=stored_object.signed_url,
            )
        finally:
            if input_video and input_video.exists():
                input_video.unlink(missing_ok=True)
            if output_video and output_video.exists():
                output_video.unlink(missing_ok=True)
            shutil.rmtree(incoming_dir, ignore_errors=True)
