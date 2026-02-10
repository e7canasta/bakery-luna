"""Domain entities (value objects and aggregates)."""

from .frame import Frame, CropInfo
from .detection import BoundingBox, Mask, Segmentation
from .pose import KeyPoint, Skeleton, PoseEstimation, COCO_KEYPOINT_NAMES
from .model_config import ModelConfig, PipelineConfig, ModelSize, ModelType, Device, Precision

__all__ = [
    # Frame
    "Frame",
    "CropInfo",
    # Detection
    "BoundingBox",
    "Mask",
    "Segmentation",
    # Pose
    "KeyPoint",
    "Skeleton",
    "PoseEstimation",
    "COCO_KEYPOINT_NAMES",
    # Config
    "ModelConfig",
    "PipelineConfig",
    "ModelSize",
    "ModelType",
    "Device",
    "Precision",
]
