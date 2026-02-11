"""
YOLO postprocessing for OpenVINO inference.

Extracted from run_lens.py - handles postprocessing of YOLO segmentation and
pose outputs including NMS, mask generation, and keypoint parsing.
"""

import cv2
import numpy as np
from typing import Tuple
from bakery_runtime.processing.geometry import xywh2xyxy, nms


def process_mask(
    protos: np.ndarray,
    mask_coef: np.ndarray,
    box: np.ndarray,
    input_shape: Tuple[int, int],
    upsample: bool = True,
) -> np.ndarray:
    """
    Generate segmentation mask from prototypes and coefficients.

    Process:
    1. Matrix multiplication: prototypes @ coefficients
    2. Sigmoid activation to [0, 1]
    3. Upsample to input resolution
    4. Crop to bounding box

    Args:
        protos: Mask prototypes [mask_dim, mask_h, mask_w] (e.g., [32, 80, 80])
        mask_coef: Mask coefficients [mask_dim] (e.g., [32])
        box: Bounding box [x1, y1, x2, y2] in input coordinates
        input_shape: (height, width) of model input
        upsample: If True, upsample mask to input_shape

    Returns:
        Binary mask [H, W] in range [0, 1]

    Example:
        >>> protos = np.random.randn(32, 80, 80)
        >>> coef = np.random.randn(32)
        >>> bbox = np.array([100, 100, 200, 200])
        >>> mask = process_mask(protos, coef, bbox, (640, 640))
        >>> mask.shape
        (640, 640)
        >>> 0 <= mask.max() <= 1
        True
    """
    # Step 1: Matrix multiplication [mask_dim] @ [mask_dim, mask_h, mask_w] -> [mask_h, mask_w]
    mask = np.einsum("i,ijk->jk", mask_coef, protos)

    # Step 2: Sigmoid activation to [0, 1]
    mask = 1 / (1 + np.exp(-mask))

    # Step 3: Upsample to input shape if needed
    if upsample:
        mask = cv2.resize(
            mask, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR
        )

    # Step 4: Crop mask to bounding box
    x1, y1, x2, y2 = box.astype(int)
    # Clamp coordinates to valid range
    x1 = max(0, min(x1, mask.shape[1] - 1))
    y1 = max(0, min(y1, mask.shape[0] - 1))
    x2 = max(0, min(x2, mask.shape[1]))
    y2 = max(0, min(y2, mask.shape[0]))

    # Create empty mask and fill only bbox region
    mask_bbox = np.zeros_like(mask)
    if x2 > x1 and y2 > y1:
        mask_bbox[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    return mask_bbox


def postprocess_segmentation(
    output_boxes: np.ndarray,
    output_masks: np.ndarray,
    input_shape: Tuple[int, int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    verbose: bool = False,
    classes: list[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Postprocess YOLO segmentation output.

    Pipeline:
    1. Parse boxes, class scores, mask coefficients
    2. Filter by confidence threshold
    3. Filter by class ID (if provided)
    4. Convert boxes from xywh to xyxy
    5. Apply NMS (Non-Maximum Suppression)
    6. Generate masks from prototypes

    Args:
        output_boxes: Detection output [1, num_classes + 4 + mask_dim, num_anchors]
                      E.g., [1, 116, 2100] = [1, 80 + 4 + 32, 2100]
        output_masks: Mask prototypes [1, mask_dim, mask_h, mask_w]
                      E.g., [1, 32, 80, 80]
        input_shape: (height, width) of model input
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        verbose: Print debug information
        classes: List of class IDs to filter by (optional)

    Returns:
        Tuple with:
        - boxes: [N, 4] in format [x1, y1, x2, y2]
        - scores: [N] confidence scores
        - class_ids: [N] class identifiers
        - masks: [N, H, W] binary masks

    Example:
        >>> output_boxes = np.random.rand(1, 116, 2100)
        >>> output_masks = np.random.rand(1, 32, 80, 80)
        >>> boxes, scores, class_ids, masks = postprocess_segmentation(
        ...     output_boxes, output_masks, (640, 640)
        ... )
    """
    # Parse output: [1, 116, 2100] -> [2100, 116]
    output_boxes = output_boxes.squeeze(0)  # Remove batch dimension

    # If features dimension is first, transpose
    if output_boxes.shape[0] < output_boxes.shape[1]:
        output_boxes = output_boxes.T  # [116, 2100] -> [2100, 116]

    # Extract components
    boxes = output_boxes[:, :4]  # Box coordinates [x, y, w, h]

    # For YOLOv11-seg: 116 = 4 (box) + 80 (classes) + 32 (mask coefs)
    num_classes = 80
    mask_dim = 32

    class_scores = output_boxes[:, 4 : 4 + num_classes]
    mask_coefs = output_boxes[:, 4 + num_classes :]

    # Verify dimensions
    if mask_coefs.shape[1] != mask_dim:
        if verbose:
            print(
                f"⚠️  Warning: Expected {mask_dim} coefficients, found {mask_coefs.shape[1]}"
            )
        mask_dim = mask_coefs.shape[1]

    # Get class IDs and scores
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]

    # Filter by confidence threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    mask_coefs = mask_coefs[mask]

    # Filter by class ID
    if classes is not None:
        mask = np.isin(class_ids, classes)
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        mask_coefs = mask_coefs[mask]

    # Early return if no detections
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Convert boxes from xywh to xyxy
    boxes = xywh2xyxy(boxes)

    # Apply NMS
    indices = nms(boxes, scores, iou_threshold)
    boxes = boxes[indices]
    scores = scores[indices]
    class_ids = class_ids[indices]
    mask_coefs = mask_coefs[indices]

    # Generate masks
    output_masks = output_masks.squeeze(0)  # [1, 32, 80, 80] -> [32, 80, 80]

    # Verify mask dimensions match
    if len(mask_coefs) > 0:
        if mask_coefs.shape[1] != output_masks.shape[0]:
            if verbose:
                print(
                    f"⚠️  Shape mismatch: mask_coefs={mask_coefs.shape}, "
                    f"output_masks={output_masks.shape}"
                )
            expected_dim = output_masks.shape[0]
            if mask_coefs.shape[1] > expected_dim:
                if verbose:
                    print(
                        f"   Truncating mask_coefs from {mask_coefs.shape[1]} to {expected_dim}"
                    )
                mask_coefs = mask_coefs[:, :expected_dim]
            else:
                return boxes, scores, class_ids, np.array([])

    # Generate mask for each detection
    masks = []
    for box, coef in zip(boxes, mask_coefs):
        mask = process_mask(output_masks, coef, box, input_shape, upsample=True)
        masks.append(mask)

    masks = np.array(masks) if masks else np.array([])

    return boxes, scores, class_ids, masks


def postprocess_pose(
    output_data: np.ndarray,
    input_shape: Tuple[int, int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Postprocess YOLO pose output.

    Pipeline:
    1. Parse boxes, person confidence, keypoints
    2. Filter by confidence threshold
    3. Convert boxes from xywh to xyxy
    4. Apply NMS
    5. Return structured keypoints [N, 17, 3]

    Args:
        output_data: Pose output [1, num_classes + 4 + num_keypoints*3, num_anchors]
                     E.g., [1, 56, 2100] = [1, 1 + 4 + 17*3, 2100] for COCO pose
        input_shape: (height, width) of model input
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS

    Returns:
        Tuple with:
        - boxes: [N, 4] in format [x1, y1, x2, y2]
        - scores: [N] person confidence scores
        - class_ids: [N] class identifiers (always 0 for person)
        - keypoints: [N, num_keypoints, 3] (x, y, confidence)

    Example:
        >>> output_data = np.random.rand(1, 56, 2100)
        >>> boxes, scores, class_ids, keypoints = postprocess_pose(
        ...     output_data, (640, 640)
        ... )
        >>> keypoints.shape[-2:]  # Should be (17, 3)
        (17, 3)
    """
    # Parse output: [1, 56, 2100] -> [2100, 56]
    output_data = output_data.squeeze(0)

    # If features dimension is first, transpose
    if output_data.shape[0] < output_data.shape[1]:
        output_data = output_data.T  # [56, 2100] -> [2100, 56]

    # Extract components
    # For COCO pose: 56 = 4 (box) + 1 (class score) + 51 (17 keypoints * 3)
    boxes = output_data[:, :4]  # Box coordinates [x, y, w, h]
    scores = output_data[:, 4]  # Person confidence

    # Keypoints: 17 points × 3 values (x, y, confidence)
    num_keypoints = 17
    keypoints_data = output_data[:, 5 : 5 + num_keypoints * 3]
    keypoints = keypoints_data.reshape(-1, num_keypoints, 3)

    # Filter by confidence threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    keypoints = keypoints[mask]

    # Early return if no detections
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Convert boxes from xywh to xyxy
    boxes = xywh2xyxy(boxes)

    # Apply NMS
    indices = nms(boxes, scores, iou_threshold)
    boxes = boxes[indices]
    scores = scores[indices]
    keypoints = keypoints[indices]

    # Class ID is always 0 (person) for pose
    class_ids = np.zeros(len(boxes), dtype=int)

    return boxes, scores, class_ids, keypoints
