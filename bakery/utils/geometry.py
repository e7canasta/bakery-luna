"""
Geometry utilities for bounding box operations.

Extracted from run_lens.py - pure functions for coordinate transformations,
NMS (Non-Maximum Suppression), and IoU (Intersection over Union) calculations.
"""

import numpy as np
from typing import List


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2] format.

    Args:
        boxes: Array of shape [..., 4] with boxes in [xc, yc, w, h] format

    Returns:
        Array of same shape with boxes in [x1, y1, x2, y2] format

    Example:
        >>> boxes = np.array([[100, 100, 50, 50]])  # center (100, 100), size 50x50
        >>> xywh2xyxy(boxes)
        array([[75., 75., 125., 125.]])  # top-left (75, 75), bottom-right (125, 125)
    """
    boxes_xyxy = np.copy(boxes)
    boxes_xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1 = xc - w/2
    boxes_xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1 = yc - h/2
    boxes_xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2 = xc + w/2
    boxes_xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2 = yc + h/2
    return boxes_xyxy


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """
    Non-Maximum Suppression (NMS).

    Filters overlapping bounding boxes, keeping only the ones with highest confidence.
    Uses greedy algorithm: iteratively select box with highest score and suppress
    all overlapping boxes with IoU > threshold.

    Args:
        boxes: Array of shape [N, 4] in format [x1, y1, x2, y2]
        scores: Array of shape [N] with confidence scores
        iou_threshold: IoU threshold for suppression (default: 0.45)

    Returns:
        List of indices of boxes to keep

    Example:
        >>> boxes = np.array([[10, 10, 50, 50], [15, 15, 55, 55], [100, 100, 150, 150]])
        >>> scores = np.array([0.9, 0.8, 0.95])
        >>> nms(boxes, scores, iou_threshold=0.5)
        [2, 0]  # Keep box 2 (highest score) and box 0 (low overlap with box 2)
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute areas of all boxes
    areas = (x2 - x1) * (y2 - y1)

    # Sort by confidence score (descending)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # Pick box with highest score
        i = order[0]
        keep.append(int(i))

        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep only boxes with IoU <= threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    IoU = Area(intersection) / Area(union)

    Args:
        bbox1: Array [x1, y1, x2, y2] for first box
        bbox2: Array [x1, y1, x2, y2] for second box

    Returns:
        IoU score in range [0, 1]

    Example:
        >>> bbox1 = np.array([10, 10, 50, 50])  # 40x40 box
        >>> bbox2 = np.array([30, 30, 70, 70])  # 40x40 box, 20x20 overlap
        >>> bbox_iou(bbox1, bbox2)
        0.142857...  # intersection=400, union=2800, iou=400/2800=0.14
    """
    # Compute intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Compute intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute areas of individual boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Compute union area
    union_area = bbox1_area + bbox2_area - inter_area

    # Compute IoU (add small epsilon to avoid division by zero)
    return inter_area / (union_area + 1e-6) if union_area > 0 else 0.0
