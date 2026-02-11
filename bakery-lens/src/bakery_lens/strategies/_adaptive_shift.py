"""
Adaptive shift lens strategy.
"""

from typing import Tuple, Optional
import supervision as sv

from ..config import FocusLensConfig
from .._types import CropInfo
from ._base import LensStrategy


class AdaptiveShiftLensStrategy:
    """
    Adaptive shift strategy.
    
    Dynamically adjusts crop position based on detection feedback.
    - Shifts window if detections are near edges.
    - Uses EMA smoothing for stability.
    - Respects frame boundaries.
    """

    def __init__(self, config: FocusLensConfig):
        self.config = config
        self.current_center_x: Optional[float] = None
        self.target_center_x: Optional[float] = None
        
        self.current_size: Optional[float] = None
        self.target_size: Optional[float] = None
        
        # Parameters
        self.edge_threshold = config.edge_threshold
        self.shift_step = config.shift_step
        self.smoothing = config.smoothing  # 0.0 = instant, 1.0 = frozen

    def compute_crop_params(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """Compute crop parameters with adaptive shift and expand."""
        base_size = self.config.focus_size
        
        # Initialize state if first frame
        if self.current_center_x is None:
            # Center
            if self.config.focus_x is None:
                initial_x = max(0, (frame_w - base_size) // 2)
            else:
                initial_x = self.config.focus_x
            
            self.current_center_x = float(initial_x + base_size / 2)
            self.target_center_x = self.current_center_x
            
            # Size
            self.current_size = float(base_size)
            self.target_size = float(base_size)

        # Apply smoothing
        self.current_center_x = (
            self.current_center_x * self.smoothing + 
            self.target_center_x * (1 - self.smoothing)
        )
        self.current_size = (
            self.current_size * self.smoothing +
            self.target_size * (1 - self.smoothing)
        )
        
        # Calculate derived values, snapped to multiples of 32 (model input requirement)
        current_w = max(32, (int(self.current_size) // 32) * 32)
        current_h = current_w
        
        # Convert center to origin (top-left)
        focus_x = int(self.current_center_x - current_w / 2)
        
        # Use configured Y or center (adaptive Y not implemented yet)
        if self.config.focus_y is None:
            focus_y = max(0, (frame_h - current_h) // 2)
        else:
            focus_y = self.config.focus_y
            
        # Clamp to frame boundaries
        # Ensure width doesn't exceed frame width
        if current_w > frame_w:
            current_w = frame_w
            focus_x = 0
        else:
            focus_x = max(0, min(focus_x, frame_w - current_w))
            
        if current_h > frame_h:
            current_h = frame_h
            focus_y = 0
        else:
             focus_y = max(0, min(focus_y, frame_h - current_h))
        
        return focus_x, focus_y, current_w, current_h

    def update(self, detections: sv.Detections) -> None:
        """
        Update target position and size based on detection feedback.
        """
        if len(detections) == 0:
            # Optional: Decay size back to base if no detections?
            # self.target_size = max(self.config.focus_size, self.target_size - 10)
            return

        current_w = int(self.current_size) if self.current_size else self.config.focus_size
        
        # Check for edge proximity
        # Detections are relative to the crop, so range is [0, current_w]
        near_left = any(box[0] < self.edge_threshold for box in detections.xyxy)
        near_right = any(box[2] > (current_w - self.edge_threshold) for box in detections.xyxy)
        
        current_target_center = self.target_center_x
        current_target_size = self.target_size
        
        step = self.shift_step

        if near_left and near_right:
            # People on both sides!
            if self.config.allow_expand:
                # Expand size
                current_target_size += step
            else:
                 # Logic conflict: Can't move left or right without losing someone.
                 # Stay put.
                 pass
        elif near_left:
            # Shift left
            current_target_center -= step
        elif near_right:
            # Shift right
            current_target_center += step
        else:
            # Nobody near edges. 
            # If we are expanded, maybe shrink back?
            if self.config.allow_expand and current_target_size > self.config.focus_size:
                # Decay size
                current_target_size -= (step / 2) # Slower decay
                current_target_size = max(current_target_size, self.config.focus_size)
            
        # Update state
        self.target_center_x = current_target_center
        self.target_size = current_target_size
