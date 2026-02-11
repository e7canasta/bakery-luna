"""
Lens strategy protocol.
"""

from typing import Protocol, Optional, TYPE_CHECKING
import supervision as sv

from .._types import CropInfo
from ..config import FocusLensConfig

# Avoid circular imports if used in future
if TYPE_CHECKING:
    pass


class LensStrategy(Protocol):
    """
    Protocol for lens cropping strategies.
    
    Strategies decide WHERE to crop.
    """
    
    def compute_crop_params(
        self, 
        frame_w: int, 
        frame_h: int, 
        frame_id: int
    ) -> CropInfo:
        """
        Compute crop parameters for the current frame.
        
        Args:
            frame_w: Original frame width
            frame_h: Original frame height
            frame_id: Frame identifier
            
        Returns:
            CropInfo with x, y, width, height, scale, padding
        """
        ...

    def update(self, detections: sv.Detections) -> None:
        """
        Update strategy state based on detection feedback.
        
        Args:
            detections: Detections from the cropped frame (crop coordinates)
        """
        ...
