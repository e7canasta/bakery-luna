"""Shared utilities."""

from .geometry import xywh2xyxy, nms, bbox_iou
from .metrics import PerformanceMetrics, FPSCounter
from .focus_lens import (
    apply_focus_lens,
    map_detections_to_full_frame,
    map_keypoints_to_full_frame,
    crop_info_to_tuple,
)

__all__ = [
    # Geometry
    "xywh2xyxy",
    "nms",
    "bbox_iou",
    # Metrics
    "PerformanceMetrics",
    "FPSCounter",
    # Focus Lens
    "apply_focus_lens",
    "map_detections_to_full_frame",
    "map_keypoints_to_full_frame",
    "crop_info_to_tuple",
]
