"""
Frame entities - represents video frames with metadata.

Domain entities for frame data structures including crop information
for focus lens functionality.
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
    """

    x: int
    y: int
    width: int
    height: int
    scale_factor: float = 1.0

    def __post_init__(self):
        """Validate crop info."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Crop dimensions must be positive, got {self.width}x{self.height}")
        if self.scale_factor <= 0:
            raise ValueError(f"Scale factor must be positive, got {self.scale_factor}")


@dataclass
class Frame:
    """
    Represents a video frame with metadata.

    Attributes:
        data: Frame pixel data in BGR format [H, W, 3]
        frame_id: Unique frame identifier (frame number)
        width: Frame width in pixels
        height: Frame height in pixels
        crop_info: Optional crop information if focus lens was applied
    """

    data: np.ndarray
    frame_id: int
    width: int
    height: int
    crop_info: Optional[CropInfo] = None

    def __post_init__(self):
        """Validate frame data."""
        if self.data.ndim != 3:
            raise ValueError(f"Frame data must be 3D array [H, W, 3], got shape {self.data.shape}")
        if self.data.shape[2] != 3:
            raise ValueError(f"Frame data must have 3 channels (BGR), got {self.data.shape[2]}")
        if self.frame_id < 0:
            raise ValueError(f"Frame ID must be non-negative, got {self.frame_id}")

    @classmethod
    def from_array(
        cls, data: np.ndarray, frame_id: int, crop_info: Optional[CropInfo] = None
    ) -> "Frame":
        """
        Create Frame from numpy array.

        Args:
            data: Frame pixel data [H, W, 3] BGR
            frame_id: Frame number
            crop_info: Optional crop information

        Returns:
            Frame instance

        Example:
            >>> frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> frame = Frame.from_array(frame_data, frame_id=0)
            >>> frame.width, frame.height
            (640, 480)
        """
        h, w = data.shape[:2]
        return cls(data=data, frame_id=frame_id, width=w, height=h, crop_info=crop_info)

    def apply_crop(self, crop_info: CropInfo) -> "Frame":
        """
        Apply crop to frame (returns new Frame - immutable operation).

        Args:
            crop_info: Crop information (x, y, width, height)

        Returns:
            New Frame with cropped data

        Example:
            >>> frame = Frame.from_array(np.zeros((480, 640, 3)), frame_id=0)
            >>> crop = CropInfo(x=100, y=100, width=320, height=320)
            >>> cropped_frame = frame.apply_crop(crop)
            >>> cropped_frame.width, cropped_frame.height
            (320, 320)
        """
        # Extract crop region
        cropped_data = self.data[
            crop_info.y : crop_info.y + crop_info.height,
            crop_info.x : crop_info.x + crop_info.width,
        ]

        return Frame(
            data=cropped_data,
            frame_id=self.frame_id,
            width=crop_info.width,
            height=crop_info.height,
            crop_info=crop_info,
        )

    @property
    def shape(self) -> tuple:
        """Get frame shape (height, width, channels)."""
        return self.data.shape

    @property
    def is_cropped(self) -> bool:
        """Check if frame has crop applied."""
        return self.crop_info is not None

    def copy(self) -> "Frame":
        """Create a deep copy of the frame."""
        return Frame(
            data=self.data.copy(),
            frame_id=self.frame_id,
            width=self.width,
            height=self.height,
            crop_info=self.crop_info,  # CropInfo is frozen, no need to copy
        )
