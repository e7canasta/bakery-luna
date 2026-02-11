"""
Internal type definitions for bakery-lens.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class CropInfo:
    """
    Immutable crop information from focus lens.
    
    Attributes:
        x: X offset of crop in original frame
        y: Y offset of crop in original frame
        width: Width of cropped region
        height: Height of cropped region
        scale_factor: Scale factor applied (for zoom strategy)
        pad_x: Padding added to X axis (total width padding)
        pad_y: Padding added to Y axis (total height padding)
    """

    x: int
    y: int
    width: int
    height: int
    scale_factor: float = 1.0
    pad_x: int = 0
    pad_y: int = 0
    
    def __post_init__(self):
        """Validate crop info."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Crop dimensions must be positive, got {self.width}x{self.height}")
        if self.scale_factor <= 0:
            raise ValueError(f"Scale factor must be positive, got {self.scale_factor}")


@dataclass(frozen=True)
class LensResult:
    """
    Result of a lens process operation.
    
    Attributes:
        frame: Cropped frame data ready for inference [H, W, 3]
        crop_info: Metadata needed for coordinate mapping
        frame_id: Frame identifier
    """
    frame: np.ndarray
    crop_info: CropInfo
    frame_id: int
