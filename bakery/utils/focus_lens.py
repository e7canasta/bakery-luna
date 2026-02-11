"""
Focus Lens utilities for crop-based inference.

DEPRECATED: Use bakery_lens.* instead.
This module re-exports functions from bakery_lens for backward compatibility.
"""

from typing import Tuple, Optional
import supervision as sv

from bakery.core.entities.frame import Frame, CropInfo as LegacyCropInfo
from bakery_lens import (
    apply_focus_lens as _apply_focus_lens,
    map_detections_to_full_frame,
    map_keypoints_to_full_frame,
    crop_info_to_tuple,
    FocusLensConfig,
    CropInfo as LensCropInfo,
)

# Re-export mapping functions directly
__all__ = [
    "apply_focus_lens",
    "map_detections_to_full_frame",
    "map_keypoints_to_full_frame",
    "crop_info_to_tuple",
]


def apply_focus_lens(
    frame: Frame,
    config: FocusLensConfig
) -> Tuple[Frame, LegacyCropInfo]:
    """
    Apply focus lens to a frame (crop-based inference).
    
    Wrapper around bakery_lens.apply_focus_lens to convert between
    bakery-luna Frame entity and numpy arrays.
    """
    cropped_data, lens_crop_info = _apply_focus_lens(
        frame.data,
        frame.frame_id,
        config
    )
    
    # Convert LensCropInfo back to LegacyCropInfo (bakery.core.entities.frame.CropInfo)
    # They should be identical in structure now
    crop_info = LegacyCropInfo(
        x=lens_crop_info.x,
        y=lens_crop_info.y,
        width=lens_crop_info.width,
        height=lens_crop_info.height,
        scale_factor=lens_crop_info.scale_factor,
        pad_x=lens_crop_info.pad_x,
        pad_y=lens_crop_info.pad_y
    )
    
    # Create cropped Frame entity
    cropped_frame = Frame.from_array(
        data=cropped_data,
        frame_id=frame.frame_id,
        crop_info=crop_info
    )
    
    return cropped_frame, crop_info
