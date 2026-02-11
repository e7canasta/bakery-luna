"""
Processing utilities for ModelInstance.
"""

from bakery_runtime.processing.image import (
    preprocess_with_metadata,
    letterbox,
    PreprocessCache,
)
from bakery_runtime.processing.results import (
    postprocess_segmentation,
    postprocess_pose,
    process_mask,
)
from bakery_runtime.processing.geometry import (
    xywh2xyxy,
    nms,
    bbox_iou,
)

__all__ = [
    # Image
    "preprocess_with_metadata",
    "letterbox",
    "PreprocessCache",
    # Results
    "postprocess_segmentation",
    "postprocess_pose",
    "process_mask",
    # Geometry
    "xywh2xyxy",
    "nms",
    "bbox_iou",
]
