from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class ZoneLayoutConfig:
    mode: str = "count_mode"
    overlap: float = 0.2
    target_zones: int = 24
    yaw_step_deg: float = 30.0
    pitch_step_deg: float = 20.0
    include_poles: bool = True
    allow_pole_center: bool = False
    max_pitch_deg: float = 85.0

    def validate(self) -> None:
        if self.mode not in {"count_mode", "step_mode"}:
            raise ValueError("mode must be `count_mode` or `step_mode`")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be within [0.0, 1.0)")
        if self.target_zones <= 0:
            raise ValueError("target_zones must be > 0")
        if self.yaw_step_deg <= 0 or self.pitch_step_deg <= 0:
            raise ValueError("yaw_step_deg and pitch_step_deg must be > 0")
        if not 0.0 < self.max_pitch_deg <= 90.0:
            raise ValueError("max_pitch_deg must be within (0, 90]")


@dataclass
class SequenceJobConfig:
    input_dir: Path
    output_dir: Path
    crop_width: int = 640
    crop_height: int = 640
    fov_x_deg: float = 90.0
    chunk_size: int = 8
    mode: str = "bilinear"
    image_ext: str = ".png"
    compression_strength: float = 0.5
    use_imagecodecs: bool = False
    writer_threads: int = 4
    use_torch_gpu: bool = False
    gpu_zone_batch_size: int = 8
    use_native_torch_backend: bool = True
    z_down: bool = False
    clip_output: bool = True
    frame_start: int = 0
    frame_end: int = -1
    layout: ZoneLayoutConfig = field(default_factory=ZoneLayoutConfig)

    def validate(self) -> None:
        self.layout.validate()
        if self.frame_start < 0:
            raise ValueError("frame_start must be >= 0")
        if self.frame_end != -1 and self.frame_end < self.frame_start:
            raise ValueError("frame_end must be >= frame_start or -1 for last frame")
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise ValueError(f"input_dir does not exist or is not a directory: {self.input_dir}")
        if self.crop_width <= 0 or self.crop_height <= 0:
            raise ValueError("crop_width and crop_height must be > 0")
        if not 1.0 <= self.fov_x_deg < 180.0:
            raise ValueError("fov_x_deg must be within [1, 180)")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.mode not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError("mode must be nearest, bilinear, or bicubic")
        if not self.image_ext.startswith("."):
            raise ValueError("image_ext should start with dot, e.g. `.png`")
        if not 0.0 <= self.compression_strength <= 1.0:
            raise ValueError("compression_strength must be within [0.0, 1.0]")
        if self.writer_threads <= 0:
            raise ValueError("writer_threads must be > 0")
        if self.gpu_zone_batch_size <= 0:
            raise ValueError("gpu_zone_batch_size must be > 0")


@dataclass
class ProgressInfo:
    total_frames: int
    frame_index: int
    frame_name: str
    total_zones: int
    zone_index: int
    message: str
    eta_seconds: Optional[float] = None
