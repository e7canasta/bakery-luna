"""
bakery_runtime.filtering — Composable post-inference filtering.

Public API:
    FilterPolicy        — Declarative filtering configuration
    filter_detections   — Apply detection-level filters (class, confidence)
    filter_keypoints    — Apply keypoint-level filters (per-keypoint confidence, min visible)
    COCO_KEYPOINTS      — Standard 17-keypoint name → index map
    resolve_keypoint_id — Accept name or index for CLI ergonomics
"""

from bakery_runtime.filtering.policy import FilterPolicy
from bakery_runtime.filtering.apply import filter_detections, filter_keypoints
from bakery_runtime.filtering.keypoints import (
    COCO_KEYPOINTS,
    COCO_KEYPOINT_NAMES,
    NUM_COCO_KEYPOINTS,
    resolve_keypoint_id,
)

__all__ = [
    "FilterPolicy",
    "filter_detections",
    "filter_keypoints",
    "COCO_KEYPOINTS",
    "COCO_KEYPOINT_NAMES",
    "NUM_COCO_KEYPOINTS",
    "resolve_keypoint_id",
]
