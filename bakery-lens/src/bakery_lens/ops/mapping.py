"""
Coordinate mapping operations.
"""

from typing import Optional, Tuple
import numpy as np
import cv2
import supervision as sv

from .._types import CropInfo


def map_detections_to_full_frame(
    detections: sv.Detections,
    crop_info: CropInfo,
    full_width: int,
    full_height: int
) -> sv.Detections:
    """
    Map detections from cropped to full frame coordinates.

    Args:
        detections: Detections in cropped frame coordinates
        crop_info: Crop information
        full_width: Full frame width
        full_height: Full frame height

    Returns:
        Detections in full frame coordinates
    """
    if len(detections) == 0:
        return detections

    # Apply inverse scale factor and offset to boxes
    scale = crop_info.scale_factor
    boxes_full = detections.xyxy.copy()

    # If zoom was applied, first undo the scaling
    if scale != 1.0:
        boxes_full = boxes_full / scale

    # Then add offset
    # Calculate padding offsets (assumes centered padding)
    pad_left = crop_info.pad_x // 2
    pad_top = crop_info.pad_y // 2

    # Add crop offset (in valid/padded space) then subtract padding to return to original space
    boxes_full[:, [0, 2]] += (crop_info.x / scale) if scale != 1.0 else (crop_info.x - pad_left)
    boxes_full[:, [1, 3]] += (crop_info.y / scale) if scale != 1.0 else (crop_info.y - pad_top)

    # Handle masks
    masks_full = None
    if detections.mask is not None and len(detections.mask) > 0:
        crop_h, crop_w = detections.mask[0].shape
        masks_full = []

        for mask in detections.mask:
            # Create full-size mask
            mask_full = np.zeros((full_height, full_width), dtype=bool)

            # If zoom was applied, resize mask first
            if scale != 1.0:
                # Calculate original crop position in unscaled coordinates
                orig_x = int(crop_info.x / scale)
                orig_y = int(crop_info.y / scale)
                orig_w = int(crop_info.width / scale)
                orig_h = int(crop_info.height / scale)

                # Resize mask to original crop size
                mask_resized = cv2.resize(
                    mask.astype(np.uint8),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                # Place in full frame
                y_end = min(orig_y + orig_h, full_height)
                x_end = min(orig_x + orig_w, full_width)
                actual_h = y_end - orig_y
                actual_w = x_end - orig_x

                mask_full[orig_y:y_end, orig_x:x_end] = mask_resized[:actual_h, :actual_w]
            else:
                # No zoom - just place mask at crop offset minus padding
                
                # Calculate placement in full frame
                start_x = crop_info.x - pad_left
                start_y = crop_info.y - pad_top
                
                # Calculate overlap between crop and full frame
                # Intersection
                inter_x1 = max(0, start_x)
                inter_y1 = max(0, start_y)
                inter_x2 = min(full_width, start_x + crop_info.width)
                inter_y2 = min(full_height, start_y + crop_info.height)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    # Offsets into the mask (crop)
                    mask_x1 = inter_x1 - start_x
                    mask_y1 = inter_y1 - start_y
                    mask_x2 = mask_x1 + (inter_x2 - inter_x1)
                    mask_y2 = mask_y1 + (inter_y2 - inter_y1)
                    
                    mask_full[inter_y1:inter_y2, inter_x1:inter_x2] = mask[mask_y1:mask_y2, mask_x1:mask_x2]

            masks_full.append(mask_full)

        masks_full = np.array(masks_full)

    # Create new Detections with mapped coordinates
    return sv.Detections(
        xyxy=boxes_full,
        confidence=detections.confidence,
        class_id=detections.class_id,
        mask=masks_full
    )


def map_keypoints_to_full_frame(
    keypoints: sv.KeyPoints,
    crop_info: CropInfo
) -> sv.KeyPoints:
    """
    Map keypoints from cropped to full frame coordinates.

    Args:
        keypoints: KeyPoints in cropped frame coordinates
        crop_info: Crop information

    Returns:
        KeyPoints in full frame coordinates

    Note:
        Keypoints at (0, 0) are treated as invisible and not offset.
    """
    if len(keypoints) == 0:
        return keypoints

    # Copy xy coordinates
    xy_full = keypoints.xy.copy()
    scale = crop_info.scale_factor

    # Create mask for valid (non-zero) keypoints
    # Keypoints at (0, 0) are invisible and should stay at origin
    valid_mask = ~np.all(np.isclose(xy_full, 0), axis=2)  # Shape: [N, num_keypoints]

    # Apply inverse scale and offset only to valid keypoints
    if scale != 1.0:
        # Undo scaling
        xy_full = np.where(
            valid_mask[:, :, np.newaxis],
            xy_full / scale,
            xy_full
        )
        # Add offset (in unscaled coordinates)
        offset_x = crop_info.x / scale
        offset_y = crop_info.y / scale
    else:
        offset_x = crop_info.x - (crop_info.pad_x // 2)
        offset_y = crop_info.y - (crop_info.pad_y // 2)

    # Apply offset to valid keypoints
    xy_full[:, :, 0] = np.where(valid_mask, xy_full[:, :, 0] + offset_x, 0)
    xy_full[:, :, 1] = np.where(valid_mask, xy_full[:, :, 1] + offset_y, 0)

    # Create new KeyPoints
    return sv.KeyPoints(
        xy=xy_full.astype(np.float32),
        confidence=keypoints.confidence,
        class_id=keypoints.class_id
    )


def crop_info_to_tuple(crop_info: Optional[CropInfo]) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert CropInfo to tuple format for annotator compatibility.

    Args:
        crop_info: CropInfo or None

    Returns:
        Tuple (x, y, width, height) or None
    """
    if crop_info is None:
        return None
    # Adjust for padding to return coordinates in original frame space
    pad_left = crop_info.pad_x // 2
    pad_top = crop_info.pad_y // 2
    return (
        crop_info.x - pad_left,
        crop_info.y - pad_top,
        crop_info.width,
        crop_info.height
    )
