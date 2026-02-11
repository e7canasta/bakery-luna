"""
Bakery Lens - Crop-based Inference Optimization.
"""

from typing import Protocol, Optional
import numpy as np
import cv2
import supervision as sv

from .config import FocusLensConfig
from ._types import CropInfo, LensResult
from .strategies import StaticLensStrategy, LensStrategy
from .ops.crop import apply_focus_lens
from .ops.mapping import map_detections_to_full_frame, map_keypoints_to_full_frame


class Lens(Protocol):
    """
    Public lens protocol.
    """
    
    def process(self, frame: np.ndarray, frame_id: int) -> LensResult:
        """Apply lens to frame. Returns cropped frame + metadata."""
        ...

    def map_detections(self, detections: sv.Detections, crop_info: CropInfo, frame_w: int, frame_h: int) -> sv.Detections:
        """Map detections from crop-space to full-frame-space."""
        ...
    
    def map_keypoints(self, keypoints: sv.KeyPoints, crop_info: CropInfo) -> sv.KeyPoints:
        """Map keypoints from crop-space to full-frame-space."""
        ...

    def update(self, detections: sv.Detections) -> None:
        """Feed detection results back for adaptive strategies."""
        ...


class BaseLens:
    """
    Base implementation of Lens protocol.
    Delegates positioning to a Strategy and cropping/mapping to Ops.
    """
    
    def __init__(self, config: FocusLensConfig, strategy: LensStrategy):
        self.config = config
        self.strategy = strategy
        
    def process(self, frame: np.ndarray, frame_id: int) -> LensResult:
        # 1. Handle frame sizing (Zoom/Pad) - currently coupled in apply_focus_lens
        # Ideally we refactor ops/crop.py to accept (x, y) from strategy.
        #
        # Let's override the config passed to apply_focus_lens with dynamic coords
        # computed by the strategy.
        
        # Calculate effective dimensions (after potential zoom/pad)
        # This duplicates logic from apply_focus_lens slightly but necessary for strategy
        frame_h, frame_w = frame.shape[:2]
        focus_size = self.config.focus_size
        eff_w, eff_h = frame_w, frame_h
        
        if frame_w < focus_size or frame_h < focus_size:
            if self.config.strategy == "zoom":
                scale = focus_size / min(frame_w, frame_h)
                eff_w = int(frame_w * scale)
                eff_h = int(frame_h * scale)
            elif self.config.strategy == "pad":
                # With pad, dimensions become at least focus_size
                eff_w = max(frame_w, focus_size)
                eff_h = max(frame_h, focus_size)
        
        # 2. Ask strategy for crop origin
        # Note: Strategy.compute_crop_origin implementation in _static.py expects effective dimensions
        crop_x, crop_y = self.strategy.compute_crop_origin(eff_w, eff_h)
        
        # 3. Create a temporary config override with the computed position
        # FocusLensConfig is frozen, so we use replace-like behavior (or just pass explicit x/y to a lower-level op)
        # 
        # Since apply_focus_lens takes config, we need a way to pass dynamic x/y.
        # Let's update apply_focus_lens signature? OR create a throwaway config.
        # Creating throwaway config is safest for now.
        
        # Creates a new config with the strategy-determined position
        # We use __dict__ copy and update because dataclass is frozen? 
        # No, frozen dataclasses support replace() but that's python 3.7+.
        # Actually FocusLensConfig is simple enough to just instantiate new one.
        
        dynamic_config = FocusLensConfig(
            focus_size=self.config.focus_size,
            focus_x=crop_x,
            focus_y=crop_y,
            strategy=self.config.strategy,
            adaptive=self.config.adaptive,
            edge_threshold=self.config.edge_threshold,
            shift_step=self.config.shift_step,
            smoothing=self.config.smoothing,
            allow_expand=self.config.allow_expand
        )
        
        # 4. Apply crop
        cropped_data, crop_info = apply_focus_lens(frame, frame_id, dynamic_config)
        
        return LensResult(
            frame=cropped_data,
            crop_info=crop_info,
            frame_id=frame_id
        )

    def map_detections(self, detections: sv.Detections, crop_info: CropInfo, frame_w: int, frame_h: int) -> sv.Detections:
        return map_detections_to_full_frame(detections, crop_info, frame_w, frame_h)

    def map_keypoints(self, keypoints: sv.KeyPoints, crop_info: CropInfo) -> sv.KeyPoints:
        return map_keypoints_to_full_frame(keypoints, crop_info)

    def update(self, detections: sv.Detections) -> None:
        self.strategy.update(detections)


def create_lens(config: FocusLensConfig) -> Lens:
    """
    Factory to create a Lens instance.
    """
    # Select strategy based on config
    # Phase 0: Always Static
    # Phase 1: Check config.adaptive -> AdaptiveShift
    
    strategy = StaticLensStrategy(config)
    return BaseLens(config, strategy)


__all__ = [
    "FocusLensConfig",
    "CropInfo",
    "LensResult",
    "Lens",
    "create_lens",
    # Re-export pure ops for backwards compatibility / low-level usage
    "apply_focus_lens", 
    "map_detections_to_full_frame",
    "map_keypoints_to_full_frame",
]
