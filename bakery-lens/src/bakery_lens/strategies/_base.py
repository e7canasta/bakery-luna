"""
Lens strategy protocol.
"""

from typing import Protocol, Optional, Tuple, TYPE_CHECKING
import supervision as sv

# Avoid circular imports if used in future
if TYPE_CHECKING:
    pass


class LensStrategy(Protocol):
    """
    Protocol for lens cropping strategies.
    
    Strategies decide WHERE and HOW LARGE to crop.
    """
    
    def compute_crop_params(
        self, 
        frame_w: int, 
        frame_h: int
    ) -> Tuple[int, int, int, int]:
        """
        Compute crop parameters (x, y, width, height).
        
        Args:
            frame_w: Effective frame width
            frame_h: Effective frame height
            
        Returns:
            Tuple (x, y, width, height)
        """
        ...

    def update(self, detections: sv.Detections) -> None:
        """
        Update strategy state based on detection feedback.
        
        Args:
            detections: Detections from the cropped frame (crop coordinates)
        """
        ...
