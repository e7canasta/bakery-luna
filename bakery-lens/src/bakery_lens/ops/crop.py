"""
Cropping operations.
"""

from typing import Tuple
import numpy as np
import cv2

from ..config import FocusLensConfig
from .._types import CropInfo


def apply_focus_lens(
    frame: np.ndarray,
    frame_id: int,
    config: FocusLensConfig
) -> Tuple[np.ndarray, CropInfo]:
    """
    Apply focus lens to a frame (crop-based inference).

    Args:
        frame: Input frame data [H, W, 3]
        frame_id: Frame identifier
        config: Focus lens configuration

    Returns:
        Tuple of (cropped_frame_data, crop_info)
    """
    frame_h, frame_w = frame.shape[:2]
    focus_size = config.focus_size
    scale_factor = 1.0
    pad_w_total = 0
    pad_h_total = 0

    frame_data = frame

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
            pad_w_total = max(0, focus_size - frame_w)
            pad_h_total = max(0, focus_size - frame_h)
            pad_left = pad_w_total // 2
            pad_right = pad_w_total - pad_left
            pad_top = pad_h_total // 2
            pad_bottom = pad_h_total - pad_top

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
        scale_factor=scale_factor,
        pad_x=pad_w_total,
        pad_y=pad_h_total
    )

    return cropped_data, crop_info
