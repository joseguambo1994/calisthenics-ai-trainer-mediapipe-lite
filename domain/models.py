from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessedVideo:
    output_path: str
    frames: int
    fps: float
