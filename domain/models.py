from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessedVideo:
    output_path: str
    frames: int
    fps: float


@dataclass(frozen=True)
class StoredObject:
    object_key: str
    object_url: str
    signed_url: str


@dataclass(frozen=True)
class ProcessedVideoDelivery:
    output_path: str
    frames: int
    fps: float
    object_key: str
    object_url: str
    video_signed_url: str
