"""
Static lens strategy.
"""

import supervision as sv

from ..config import FocusLensConfig
from .._types import CropInfo
from ._base import LensStrategy


class StaticLensStrategy:
    """
    Static cropping strategy.
    
    Strictly follows configuration (fixed position or centered).
    Ignores detection feedback.
    """

    def __init__(self, config: FocusLensConfig):
        self.config = config

    def compute_crop_params(
        self, 
        frame_w: int, 
        frame_h: int, 
        frame_id: int
    ) -> CropInfo:
        """Compute static crop parameters."""
        focus_size = self.config.focus_size
        
        # Calculate padding if needed (strategy="pad")
        pad_w_total = 0
        pad_h_total = 0
        scale_factor = 1.0
        
        # Determine effective frame dimensions for positioning
        eff_w, eff_h = frame_w, frame_h
        
        if frame_w < focus_size or frame_h < focus_size:
            if self.config.strategy == "zoom":
                 # Zoom logic handled by ops/crop.py, but we need scale factor here 
                 # to report correct CropInfo? 
                 # Actually, CropInfo is fully constructed in ops/crop.py right now.
                 # 
                 # REFACTOR NOTICE: 
                 # ops/crop.py does BOTH calculation and cropping currently.
                 # To support strategy pattern properly, we should split:
                 # 1. Strategy computes params (x, y, scale, pad)
                 # 2. Ops applies crop using params
                 #
                 # However, for Phase 0 migration, let's keep the logic in ops/crop.py
                 # and make the Strategy just a wrapper or configuration holder if possible.
                 #
                 # BUT, the architecture plan says Lens.process() -> (Frame, CropInfo).
                 # If we want strategies to control x/y, they effectively need to replicate
                 # the positioning logic currently in apply_focus_lens.
                 
                 pass
            elif self.config.strategy == "pad":
                pass
        
        # ... logic duplication risk ...
        
        # For Phase 0, we can implementation Lens.process() by delegating to ops/crop.py directly
        # since StaticLens is just the "default" behavior already encoded there.
        #
        # But to prepare for Adaptive, we need the split.
        #
        # Let's implement calculate_crop_window here.
        
        return CropInfo(0,0,0,0)  # Placeholder, see implementation implementation below
"""

RE-EVALUATION:
`ops/crop.py` currently holds all the logic including:
1. Scaling/Padding (Zoom/Pad strategy)
2. Positioning (Center vs Explicit)

To support Adaptive Strategy:
- Scaling/Padding should likely remain a "pre-step".
- Positioning (x,y) is what Adaptive Strategy changes.

So:
Lens.process():
    1. Handle frame size (Zoom/Pad) -> eff_frame, scale, pads
    2. Strategy.compute_crop_origin(eff_w, eff_h) -> x, y
    3. Extract crop
    4. Return result

Let's adjust StaticLensStrategy to just provide (x, y).

"""

class StaticLensStrategy:
    """
    Static cropping strategy.
    
    Returns configured or centered position.
    """

    def __init__(self, config: FocusLensConfig):
        self.config = config

    def compute_crop_origin(self, frame_w: int, frame_h: int) -> tuple[int, int]:
        """
        Compute top-left crop origin (x, y).
        
        Args:
            frame_w: Width of frame (after any scaling/padding)
            frame_h: Height of frame (after any scaling/padding)
        """
        focus_size = self.config.focus_size
        
        # Calculate X
        if self.config.focus_x is None:
            focus_x = max(0, (frame_w - focus_size) // 2)
        else:
            focus_x = self.config.focus_x
            
        # Calculate Y
        if self.config.focus_y is None:
            focus_y = max(0, (frame_h - focus_size) // 2)
        else:
            focus_y = self.config.focus_y
            
        # Clamp (always safe)
        focus_x = max(0, min(focus_x, frame_w - focus_size))
        focus_y = max(0, min(focus_y, frame_h - focus_size))
        
        return focus_x, focus_y

    def update(self, detections: sv.Detections) -> None:
        """No-op for static strategy."""
        pass
