from pathlib import Path
from uuid import uuid4

from domain.ports import PoseVideoProcessor, TelegramVideoGateway


class ProcessTelegramVideoUseCase:
    def __init__(
        self,
        gateway: TelegramVideoGateway,
        processor: PoseVideoProcessor,
        workspace_dir: Path,
    ) -> None:
        self._gateway = gateway
        self._processor = processor
        self._workspace_dir = workspace_dir

    def execute(self, file_id: str):
        request_id = uuid4().hex
        incoming_dir = self._workspace_dir / "incoming" / request_id
        processed_dir = self._workspace_dir / "processed"

        incoming_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        input_video = self._gateway.download_video(file_id=file_id, target_dir=incoming_dir)
        output_video = processed_dir / f"{input_video.stem}-processed.mp4"

        return self._processor.process(input_video=input_video, output_video=output_video)
