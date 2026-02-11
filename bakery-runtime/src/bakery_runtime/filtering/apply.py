"""
Stateless filter functions for post-inference results.

Pure numpy operations — no domain entity dependency.
Designed to run post-NMS, pre-entity construction (earliest useful point).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from bakery_runtime.filtering.policy import FilterPolicy


def filter_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    masks: np.ndarray,
    policy: FilterPolicy,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply detection-level filters: class whitelist + per-class confidence.

    Pipeline:
    1. Class whitelist filter (keep only policy.classes)
    2. Per-class confidence filter (each class_id checked against its resolved threshold)

    Args:
        boxes: [N, 4] bounding boxes (xyxy).
        scores: [N] detection confidence scores.
        class_ids: [N] class identifiers.
        masks: [N, H, W] segmentation masks (or empty array).
        policy: FilterPolicy with filtering rules.

    Returns:
        Filtered (boxes, scores, class_ids, masks) — same shapes, fewer rows.

    Example:
        >>> policy = FilterPolicy(classes=(0,), per_class_confidence={0: 0.5})
        >>> boxes_f, scores_f, cls_f, masks_f = filter_detections(
        ...     boxes, scores, class_ids, masks, policy
        ... )
    """
    if len(boxes) == 0:
        return boxes, scores, class_ids, masks

    keep = np.ones(len(boxes), dtype=bool)

    # 1. Class whitelist
    if policy.has_class_filter:
        keep &= np.isin(class_ids, policy.classes)

    # 2. Per-class confidence thresholding
    if policy.has_per_class_confidence:
        for cls_id, threshold in policy.per_class_confidence.items():
            cls_mask = class_ids == cls_id
            keep &= ~cls_mask | (scores >= threshold)

    # Apply mask
    if not keep.all():
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        if len(masks) > 0:
            masks = masks[keep]

    return boxes, scores, class_ids, masks


def filter_keypoints(
    keypoints: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    policy: FilterPolicy,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply keypoint-level filters.

    Pipeline:
    1. Zero out keypoints below their resolved confidence threshold
    2. Optionally drop entire skeletons if fewer than keypoint_min_visible are visible

    Args:
        keypoints: [N, 17, 3] array of (x, y, confidence) per keypoint.
        boxes: [N, 4] bounding boxes for each skeleton.
        scores: [N] detection scores for each skeleton.
        policy: FilterPolicy with keypoint filtering rules.

    Returns:
        Filtered (keypoints, boxes, scores).
        Keypoints below threshold have their coordinates zeroed out.
        Skeletons below min_visible count are removed entirely.
    """
    if len(keypoints) == 0:
        return keypoints, boxes, scores

    num_kp = keypoints.shape[1]
    filtered_kp = keypoints.copy()

    # 1. Per-keypoint confidence thresholding
    for kp_id in range(num_kp):
        threshold = policy.resolve_keypoint_confidence(kp_id)
        below = filtered_kp[:, kp_id, 2] < threshold
        # Zero out coordinates for keypoints below threshold
        filtered_kp[below, kp_id, :] = 0.0

    # 2. Drop skeletons with too few visible keypoints
    if policy.keypoint_min_visible is not None:
        visible_count = np.sum(filtered_kp[:, :, 2] > 0, axis=1)
        keep = visible_count >= policy.keypoint_min_visible

        if not keep.all():
            filtered_kp = filtered_kp[keep]
            boxes = boxes[keep]
            scores = scores[keep]

    return filtered_kp, boxes, scores
