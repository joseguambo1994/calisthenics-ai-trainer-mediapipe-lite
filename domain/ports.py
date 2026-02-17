from pathlib import Path
from typing import Protocol

from domain.models import LandmarksGenerationResult, ProcessedVideo, StoredObject


class TelegramVideoGateway(Protocol):
    def download_video(self, file_id: str, target_dir: Path) -> Path:
        ...


class PoseVideoProcessor(Protocol):
    def process(self, input_video: Path, output_video: Path) -> ProcessedVideo:
        ...


class ObjectStorageGateway(Protocol):
    def upload_file(
        self,
        source_path: Path,
        object_key: str,
        content_type: str | None = None,
    ) -> StoredObject:
        ...


class MovementLandmarksGenerator(Protocol):
    def generate(self) -> LandmarksGenerationResult:
        ...
