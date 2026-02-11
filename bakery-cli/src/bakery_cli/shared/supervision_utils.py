"""
Shared utilities for supervision entities conversion.
"""
import numpy as np
import cv2
import supervision as sv
from typing import Tuple, Dict

def convert_to_supervision_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    masks: np.ndarray,
    metadata: Dict
) -> sv.Detections:
    """
    Convert model outputs to supervision.Detections format.
    Handles coordinate mapping (undoing letterbox).
    """
    if len(boxes) == 0:
        return sv.Detections.empty()

    ratio = metadata["ratio"]
    pad_w = metadata["pad_w"]
    pad_h = metadata["pad_h"]
    orig_h = metadata["orig_h"]
    orig_w = metadata["orig_w"]

    # 1. Undo letterbox from boxes
    boxes_orig = boxes.copy()
    boxes_orig[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio
    boxes_orig[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio

    # Clamp to original dimensions
    boxes_orig[:, [0, 2]] = np.clip(boxes_orig[:, [0, 2]], 0, orig_w)
    boxes_orig[:, [1, 3]] = np.clip(boxes_orig[:, [1, 3]], 0, orig_h)

    # 2. Undo letterbox from masks
    masks_orig = []
    if masks is not None and len(masks) > 0:
        for mask in masks:
            mask_h, mask_w = mask.shape
            pad_top = int(pad_h)
            pad_bottom = int(pad_h)
            pad_left = int(pad_w)
            pad_right = int(pad_w)

            # Crop mask
            mask_unpadded = mask[pad_top:mask_h-pad_bottom, pad_left:mask_w-pad_right]

            # Resize
            if mask_unpadded.shape[0] > 0 and mask_unpadded.shape[1] > 0:
                mask_resized = cv2.resize(mask_unpadded, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            else:
                mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            mask_bin = (mask_resized > 0.5).astype(bool)
            masks_orig.append(mask_bin)
        masks_orig = np.array(masks_orig)
    else:
        masks_orig = None

    # 3. Create sv.Detections
    detections = sv.Detections(
        xyxy=boxes_orig,
        confidence=scores,
        class_id=class_ids.astype(int),
        mask=masks_orig
    )

    return detections


def convert_to_supervision_keypoints(
    keypoints: np.ndarray,
    metadata: Dict,
    keypoint_conf_threshold: float = 0.5
) -> sv.KeyPoints:
    """
    Convert model outputs to supervision.KeyPoints format.
    Handles coordinate mapping (undoing letterbox).
    """
    if len(keypoints) == 0:
        return sv.KeyPoints.empty()

    ratio = metadata["ratio"]
    pad_w = metadata["pad_w"]
    pad_h = metadata["pad_h"]
    orig_h = metadata["orig_h"]
    orig_w = metadata["orig_w"]

    # Extract xy and confidence
    xy = keypoints[:, :, :2].copy()
    confidence = keypoints[:, :, 2].copy()

    low_conf_mask = confidence < keypoint_conf_threshold

    # Undo letterbox
    xy[:, :, 0] = (xy[:, :, 0] - pad_w) / ratio
    xy[:, :, 1] = (xy[:, :, 1] - pad_h) / ratio

    # Clamp
    xy[:, :, 0] = np.clip(xy[:, :, 0], 0, orig_w)
    xy[:, :, 1] = np.clip(xy[:, :, 1], 0, orig_h)

    # Mask low confidence
    xy[low_conf_mask] = 0
    confidence[low_conf_mask] = 0

    return sv.KeyPoints(
        xy=xy.astype(np.float32),
        confidence=confidence.astype(np.float32),
        class_id=np.zeros(len(keypoints), dtype=int)
    )
