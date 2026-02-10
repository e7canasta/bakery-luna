"""
Focus Lens utilities for crop-based inference.

This module provides functions for applying a focus lens (square crop)
to frames and mapping inference results back to full frame coordinates.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
import supervision as sv

from bakery.core.entities.frame import Frame, CropInfo
from bakery.core.entities.focus_lens_config import FocusLensConfig


def apply_focus_lens(
    frame: Frame,
    config: FocusLensConfig
) -> Tuple[Frame, CropInfo]:
    """
    Apply focus lens to a frame (crop-based inference).

    Extracts a square region from the frame before inference,
    concentrating pixels on the region of interest.

    Args:
        frame: Input frame
        config: Focus lens configuration

    Returns:
        Tuple of (cropped_frame, crop_info)
        - cropped_frame: Frame with cropped data
        - crop_info: CropInfo with offset and scale information

    Example:
        >>> config = FocusLensConfig(focus_size=640)
        >>> cropped, crop_info = apply_focus_lens(frame, config)
        >>> # cropped.width == 640, cropped.height == 640
    """
    frame_data = frame.data
    frame_h, frame_w = frame_data.shape[:2]
    focus_size = config.focus_size
    scale_factor = 1.0

    # Handle frames smaller than focus_size
    if frame_w < focus_size or frame_h < focus_size:
        if config.strategy == "zoom":
            # Scale up frame to at least focus_size
            scale = focus_size / min(frame_w, frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            frame_data = cv2.resize(
                frame_data, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            frame_h, frame_w = new_h, new_w
            scale_factor = scale

        elif config.strategy == "pad":
            # Pad frame to focus_size
            pad_w = max(0, focus_size - frame_w)
            pad_h = max(0, focus_size - frame_h)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            frame_data = cv2.copyMakeBorder(
                frame_data,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)  # Black padding
            )
            frame_h, frame_w = frame_data.shape[:2]

    # Calculate crop position
    if config.focus_x is None:
        focus_x = max(0, (frame_w - focus_size) // 2)
    else:
        focus_x = config.focus_x

    if config.focus_y is None:
        focus_y = max(0, (frame_h - focus_size) // 2)
    else:
        focus_y = config.focus_y

    # Clamp to frame boundaries
    focus_x = max(0, min(focus_x, frame_w - focus_size))
    focus_y = max(0, min(focus_y, frame_h - focus_size))

    # Extract crop
    cropped_data = frame_data[
        focus_y:focus_y + focus_size,
        focus_x:focus_x + focus_size
    ]

    # Create CropInfo
    crop_info = CropInfo(
        x=focus_x,
        y=focus_y,
        width=focus_size,
        height=focus_size,
        scale_factor=scale_factor
    )

    # Create cropped Frame
    cropped_frame = Frame.from_array(
        data=cropped_data,
        frame_id=frame.frame_id,
        crop_info=crop_info
    )

    return cropped_frame, crop_info


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
        crop_info: Crop information from apply_focus_lens
        full_width: Full frame width
        full_height: Full frame height

    Returns:
        Detections in full frame coordinates

    Example:
        >>> # Detection at (10, 10) in crop
        >>> # Crop at (100, 100) in full frame
        >>> # Result: Detection at (110, 110) in full frame
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
    boxes_full[:, [0, 2]] += crop_info.x / scale if scale != 1.0 else crop_info.x
    boxes_full[:, [1, 3]] += crop_info.y / scale if scale != 1.0 else crop_info.y

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
                # No zoom - just place mask at crop offset
                y_end = min(crop_info.y + crop_h, full_height)
                x_end = min(crop_info.x + crop_w, full_width)
                actual_h = y_end - crop_info.y
                actual_w = x_end - crop_info.x

                mask_full[crop_info.y:y_end, crop_info.x:x_end] = mask[:actual_h, :actual_w]

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
        crop_info: Crop information from apply_focus_lens

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
        offset_x = crop_info.x
        offset_y = crop_info.y

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
    return (crop_info.x, crop_info.y, crop_info.width, crop_info.height)
