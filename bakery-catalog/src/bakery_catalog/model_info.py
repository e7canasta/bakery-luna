"""
Bakery Catalog - Model Info
============================

Domain entity describing a model artifact on disk.
This is a passive data object — it does not execute anything.

ModelInfo is produced by ModelRepository and consumed by bakery-runtime
to create executable ModelInstance objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ModelType(Enum):
    """Model task type."""

    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    POSE = "pose"


class Precision(Enum):
    """Model precision / quantization level."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT8_CALIBRATED = "int8_calibrated"


class ModelSize(Enum):
    """YOLO model size variant."""

    NANO = "n"
    SMALL = "s"
    MEDIUM = "m"
    LARGE = "l"
    XLARGE = "x"


@dataclass(frozen=True)
class ModelInfo:
    """
    Immutable description of a model artifact on disk.

    This is a catalog-level entity that describes WHAT a model is
    and WHERE it lives, without any runtime concerns (device, confidence, etc).

    Attributes:
        model_path: Path to the model file (.xml or .onnx)
        model_name: Full model name (e.g. "yolo26n-seg")
        model_type: Task type (segmentation, pose, detection)
        resolution: Input resolution in pixels (e.g. 320)
        precision: Model precision (fp16, int8, etc.)
        yolo_version: YOLO version string (e.g. "26")
        model_size: Size variant (n, s, m, l, x)
        metadata: Additional metadata extracted from the model file
    """

    model_path: Path
    model_name: str
    model_type: ModelType
    resolution: int
    precision: Precision
    yolo_version: str = "11"
    model_size: ModelSize = ModelSize.NANO
    metadata: dict = field(default_factory=dict)

    @property
    def model_dir(self) -> Path:
        """Directory containing the model file."""
        return self.model_path.parent

    @property
    def has_bin(self) -> bool:
        """Check if companion .bin file exists (OpenVINO models)."""
        return self.model_path.with_suffix(".bin").exists()

    @property
    def is_openvino(self) -> bool:
        """Check if this is an OpenVINO model (.xml)."""
        return self.model_path.suffix == ".xml"

    @property
    def is_onnx(self) -> bool:
        """Check if this is an ONNX model."""
        return self.model_path.suffix == ".onnx"

    def exists(self) -> bool:
        """Check if the model file physically exists on disk."""
        if self.is_openvino:
            return self.model_path.exists() and self.has_bin
        return self.model_path.exists()
