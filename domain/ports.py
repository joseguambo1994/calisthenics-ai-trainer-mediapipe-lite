from pathlib import Path
from typing import Protocol

from domain.models import ProcessedVideo


class TelegramVideoGateway(Protocol):
    def download_video(self, file_id: str, target_dir: Path) -> Path:
        ...


class PoseVideoProcessor(Protocol):
    def process(self, input_video: Path, output_video: Path) -> ProcessedVideo:
        ...
