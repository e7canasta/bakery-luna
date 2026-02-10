"""
Image preprocessing for YOLO models.

Provides functions for image preprocessing including:
- Letterbox resizing (maintaining aspect ratio with padding)
- BGR to RGB conversion
- Normalization and scaling
- Channel transposition (HWC to CHW)
- Caching for efficiency
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass, field
import numpy as np
import cv2


@dataclass
class PreprocessMetadata:
    """Metadata from preprocessing operation."""
    scale_ratio: float  # Resize ratio applied
    pad_width: float    # Horizontal padding applied
    pad_height: float   # Vertical padding applied


@dataclass
class PreprocessCache:
    """
    Cache for preprocessing results to avoid redundant computation.
    
    Caches preprocessed images per (frame_id, target_shape) key.
    Automatically invalidates when frame_id changes to next sequential frame.
    """
    
    _cache: Dict[Tuple[int, Tuple[int, int]], Tuple[np.ndarray, PreprocessMetadata]] = field(
        default_factory=dict
    )
    _last_frame_id: Optional[int] = field(default=None)
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)
    
    def get_or_compute(
        self,
        frame_data: np.ndarray,
        frame_id: int,
        target_shape: Tuple[int, int],
        compute_fn=None
    ) -> Tuple[np.ndarray, PreprocessMetadata]:
        """
        Get preprocessed result from cache or compute if missing.
        
        Args:
            frame_data: Frame pixel data [H, W, 3]
            frame_id: Frame identifier
            target_shape: Target shape (height, width)
            compute_fn: Function to compute if cache miss
            
        Returns:
            Tuple of (preprocessed_array, metadata)
        """
        # Invalidate cache on new frame (video frames are sequential)
        if self._last_frame_id is not None and frame_id != self._last_frame_id:
            self.clear()
        self._last_frame_id = frame_id
        
        key = (frame_id, target_shape)
        
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        else:
            self.misses += 1
            if compute_fn is None:
                compute_fn = preprocess_with_metadata
            
            result = compute_fn(frame_data, target_shape)
            self._cache[key] = result
            return result
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {"hits": self.hits, "misses": self.misses}
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize image while maintaining aspect ratio using letterbox padding.
    
    Maintains aspect ratio by:
    1. Computing scale ratio based on smaller dimension
    2. Resizing image proportionally
    3. Adding padding to match target shape
    
    Args:
        img: Input image [H, W, 3] in BGR format
        new_shape: Target shape (height, width)
        color: RGB padding color (will be converted to BGR)
        
    Returns:
        Tuple of:
        - Processed image [new_h, new_w, 3]
        - Scale ratio applied
        - Padding tuple (pad_w, pad_h) in pixels
        
    Example:
        >>> img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        >>> result, ratio, (pad_w, pad_h) = letterbox(img, new_shape=(640, 640))
        >>> result.shape
        (640, 640, 3)
    """
    shape = img.shape[:2]  # current shape [height, width]
    h, w = shape
    
    # Scale ratio (new / old) - use minimum to fit within new_shape
    r = min(new_shape[0] / h, new_shape[1] / w)
    
    # Compute new unpadded dimensions
    new_unpad = (
        int(round(w * r)),
        int(round(h * r))
    )
    
    # Compute padding (split evenly on both sides)
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw /= 2  # divide padding equally
    dh /= 2
    
    # Resize if needed
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Apply padding
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    
    # Note: color tuple needs to match image format (BGR)
    bgr_color = (color[2], color[1], color[0]) if len(color) == 3 else color
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=bgr_color
    )
    
    return img, r, (dw, dh)


def preprocess(
    img: np.ndarray,
    input_shape: Tuple[int, int] = (640, 640)
) -> np.ndarray:
    """
    Preprocess image for YOLO inference.
    
    Steps:
    1. Letterbox resize maintaining aspect ratio
    2. Convert BGR to RGB
    3. Normalize to [0, 1] float32
    4. Transpose HWC to CHW
    5. Add batch dimension
    
    Args:
        img: Input image [H, W, 3] in BGR format (OpenCV)
        input_shape: Target shape (height, width) for YOLO model
        
    Returns:
        Preprocessed tensor [1, 3, H, W] in float32 format
        
    Example:
        >>> img = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> tensor = preprocess(img, input_shape=(640, 640))
        >>> tensor.shape
        (1, 3, 640, 640)
        >>> tensor.dtype
        dtype('float32')
    """
    # Letterbox resize
    img_letterbox, _, _ = letterbox(img, new_shape=input_shape)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # HWC to CHW
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    
    # Add batch dimension
    img_batch = np.expand_dims(img_chw, axis=0)
    
    return img_batch


def preprocess_with_metadata(
    img: np.ndarray,
    input_shape: Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, PreprocessMetadata]:
    """
    Preprocess image for YOLO inference, returning metadata for post-processing.
    
    Metadata is needed to map inference results back to original image coordinates.
    
    Steps:
    1. Letterbox resize maintaining aspect ratio (get scale ratio and padding)
    2. Convert BGR to RGB
    3. Normalize to [0, 1] float32
    4. Transpose HWC to CHW
    5. Add batch dimension
    
    Args:
        img: Input image [H, W, 3] in BGR format (OpenCV)
        input_shape: Target shape (height, width) for YOLO model
        
    Returns:
        Tuple of:
        - Preprocessed tensor [1, 3, H, W] in float32
        - PreprocessMetadata with scale_ratio and padding info
        
    Example:
        >>> img = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> tensor, metadata = preprocess_with_metadata(img, input_shape=(640, 640))
        >>> tensor.shape
        (1, 3, 640, 640)
        >>> metadata.scale_ratio
        0.8333...
    """
    # Letterbox resize - get metadata
    img_letterbox, scale_ratio, (pad_w, pad_h) = letterbox(img, new_shape=input_shape)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # HWC to CHW
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    
    # Add batch dimension
    img_batch = np.expand_dims(img_chw, axis=0)
    
    # Create metadata
    metadata = PreprocessMetadata(
        scale_ratio=scale_ratio,
        pad_width=pad_w,
        pad_height=pad_h
    )
    
    return img_batch, metadata
