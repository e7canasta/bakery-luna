"""
Configuration entities - represents model and pipeline configurations.

Domain entities for configuring models (size, resolution, device, precision)
and pipelines (segmentation + pose configurations, smart scheduling parameters).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from pathlib import Path


class ModelSize(Enum):
    """YOLO model size variants."""

    SMALL = "s"
    MEDIUM = "m"
    LARGE = "l"
    XLARGE = "x"


class ModelType(Enum):
    """Model type - segmentation or pose estimation."""

    SEGMENTATION = "segmentation"
    POSE = "pose"


class Device(Enum):
    """Inference device options."""

    GPU = "GPU"
    CPU = "CPU"
    AUTO = "AUTO"


class Precision(Enum):
    """Model precision options."""

    FP16 = "fp16"
    FP32 = "fp32"
    INT8 = "int8"


@dataclass
class ModelConfig:
    """
    Configuration for a single model (segmentation or pose).

    Attributes:
        model_path: Path to the OpenVINO model file (.xml)
        model_type: Type of model (segmentation or pose)
        resolution: Input resolution (must be multiple of 32)
        device: Inference device (GPU, CPU, AUTO)
        precision: Model precision (fp16, fp32, int8)
        confidence: Confidence threshold for detections
        model_size: Model size variant (s, m, l, x) - optional for dynamic models
    """

    model_path: Path
    model_type: ModelType
    resolution: int
    device: Device
    precision: Precision
    confidence: float
    model_size: Optional[ModelSize] = None

    def __post_init__(self):
        """Validate model configuration."""
        # Validate resolution is multiple of 32
        if self.resolution % 32 != 0:
            raise ValueError(
                f"Resolution must be multiple of 32, got {self.resolution}"
            )

        # Validate resolution is in reasonable range
        if not (160 <= self.resolution <= 1280):
            raise ValueError(
                f"Resolution must be in range [160, 1280], got {self.resolution}"
            )

        # Validate confidence threshold
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"Confidence must be in range [0, 1], got {self.confidence}"
            )

    @property
    def input_shape(self) -> tuple:
        """Get model input shape (batch, channels, height, width)."""
        return (1, 3, self.resolution, self.resolution)

    @classmethod
    def from_args(
        cls,
        model_path: Path,
        model_type: str,
        resolution: int,
        device: str,
        precision: str,
        confidence: float,
        model_size: Optional[str] = None,
    ) -> "ModelConfig":
        """
        Create ModelConfig from string arguments.

        Args:
            model_path: Path to model file
            model_type: Model type ("segmentation", "pose")
            resolution: Input resolution
            device: Device ("GPU", "CPU", "AUTO")
            precision: Precision ("fp16", "fp32", "int8")
            confidence: Confidence threshold
            model_size: Optional model size ("s", "m", "l", "x")

        Returns:
            ModelConfig instance

        Example:
            >>> config = ModelConfig.from_args(
            ...     Path("model.xml"), "segmentation", 256, "GPU", "fp16", 0.25, "l"
            ... )
        """
        return cls(
            model_path=model_path,
            model_type=ModelType(model_type),
            resolution=resolution,
            device=Device(device),
            precision=Precision(precision),
            confidence=confidence,
            model_size=ModelSize(model_size) if model_size else None,
        )


@dataclass
class PipelineConfig:
    """
    Configuration for the entire dual-model pipeline.

    Attributes:
        segmentation: Segmentation model configuration
        pose: Pose model configuration
        seg_interval: Run segmentation every N frames (smart scheduling)
        filter_keypoints: Enable keypoint filtering with masks
        keypoint_match_strategy: Strategy for matching keypoints to masks
            ("nearest", "overlap", "dynamic")
        overlap_threshold: IoU threshold for dynamic strategy
        confidence_threshold: Minimum confidence for detections
        class_filter: Optional list of class IDs to keep (None = all classes)
    """

    segmentation: ModelConfig
    pose: ModelConfig
    seg_interval: int = 5
    filter_keypoints: bool = True
    keypoint_match_strategy: str = "dynamic"
    overlap_threshold: float = 0.3
    confidence_threshold: float = 0.5
    class_filter: Optional[List[int]] = None

    def __post_init__(self):
        """Validate pipeline configuration."""
        # Validate seg_interval
        if self.seg_interval < 1:
            raise ValueError(f"seg_interval must be >= 1, got {self.seg_interval}")

        # Validate strategy
        valid_strategies = ["nearest", "overlap", "dynamic"]
        if self.keypoint_match_strategy not in valid_strategies:
            raise ValueError(
                f"keypoint_match_strategy must be one of {valid_strategies}, "
                f"got {self.keypoint_match_strategy}"
            )

        # Validate thresholds
        if not 0 <= self.overlap_threshold <= 1:
            raise ValueError(
                f"overlap_threshold must be in [0, 1], got {self.overlap_threshold}"
            )

        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )

    @property
    def is_same_resolution(self) -> bool:
        """
        Check if segmentation and pose models use same resolution.

        Important for preprocessing cache optimization - if resolutions match,
        we only need to preprocess once.
        """
        return self.segmentation.resolution == self.pose.resolution

    def get_summary(self) -> dict:
        """
        Get configuration summary as dictionary.

        Returns:
            Dictionary with all configuration parameters
        """
        return {
            "segmentation": {
                "model_size": self.segmentation.model_size.value,
                "resolution": self.segmentation.resolution,
                "device": self.segmentation.device.value,
                "precision": self.segmentation.precision.value,
            },
            "pose": {
                "model_size": self.pose.model_size.value,
                "resolution": self.pose.resolution,
                "device": self.pose.device.value,
                "precision": self.pose.precision.value,
            },
            "pipeline": {
                "seg_interval": self.seg_interval,
                "filter_keypoints": self.filter_keypoints,
                "keypoint_match_strategy": self.keypoint_match_strategy,
                "overlap_threshold": self.overlap_threshold,
                "confidence_threshold": self.confidence_threshold,
                "class_filter": self.class_filter,
            },
        }

    def print_summary(self):
        """Print formatted configuration summary."""
        print("\n🔧 Pipeline Configuration")
        print("\n📦 Segmentation Model:")
        print(f"   Size: {self.segmentation.model_size.value}")
        print(f"   Resolution: {self.segmentation.resolution}px")
        print(f"   Device: {self.segmentation.device.value}")
        print(f"   Precision: {self.segmentation.precision.value}")

        print("\n🦴 Pose Model:")
        print(f"   Size: {self.pose.model_size.value}")
        print(f"   Resolution: {self.pose.resolution}px")
        print(f"   Device: {self.pose.device.value}")
        print(f"   Precision: {self.pose.precision.value}")

        print("\n⚙️  Pipeline Settings:")
        print(f"   Seg interval: {self.seg_interval} frames")
        print(f"   Filter keypoints: {self.filter_keypoints}")
        if self.filter_keypoints:
            print(f"   Match strategy: {self.keypoint_match_strategy}")
            print(f"   Overlap threshold: {self.overlap_threshold}")
        print(f"   Confidence threshold: {self.confidence_threshold}")
        if self.class_filter:
            print(f"   Class filter: {self.class_filter}")
