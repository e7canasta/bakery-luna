import numpy as np
import cv2
from typing import Tuple, Dict

def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image maintaining aspect ratio (letterbox).
    
    Returns:
        img: Resized image with padding
        ratio: Resize ratio
        (dw, dh): Applied padding
    """
    shape = img.shape[:2]  # current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    return img, r, (dw, dh)


def preprocess_with_metadata(
    img: np.ndarray,
    input_shape: Tuple[int, int]
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Pre-process image for YOLO segmentation inference.
    
    Args:
        img: BGR Image (OpenCV)
        input_shape: (height, width) of model input
    
    Returns:
        Tuple with:
        - Tensor ready for inference [1, 3, H, W]
        - ratio: Scale ratio applied
        - (pad_w, pad_h): Padding applied on each axis
    """
    # Letterbox resize
    img_resized, ratio, (dw, dh) = letterbox(img, new_shape=input_shape)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_norm = img_rgb.astype(np.float32) / 255.0
    
    # HWC to CHW
    img_chw = np.transpose(img_norm, (2, 0, 1))
    
    # Add batch dimension
    img_batch = np.expand_dims(img_chw, axis=0)
    
    return img_batch, ratio, (dw, dh)


class PreprocessCache:
    """
    Cache preprocessing results to avoid redundant computation.

    Key optimization: If seg_shape == pose_shape, compute once and reuse.
    Otherwise, compute separately and cache both.
    """

    def __init__(self):
        self.seg_tensor = None
        self.seg_metadata = None
        self.pose_tensor = None
        self.pose_metadata = None
        self.last_frame_id = -1

    def get_or_compute(
        self,
        frame: np.ndarray,
        frame_id: int,
        seg_shape: Tuple[int, int],
        pose_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, Dict, np.ndarray, Dict]:
        """
        Get cached tensors or compute if needed.

        Args:
            frame: Input frame (BGR, OpenCV format)
            frame_id: Current frame ID
            seg_shape: (height, width) for segmentation model
            pose_shape: (height, width) for pose model

        Returns:
            Tuple with:
            - seg_tensor: Preprocessed tensor for segmentation [1, 3, H, W]
            - seg_metadata: Dict with {ratio, pad_w, pad_h}
            - pose_tensor: Preprocessed tensor for pose [1, 3, H, W]
            - pose_metadata: Dict with {ratio, pad_w, pad_h}
        """
        # Get original frame dimensions
        orig_h, orig_w = frame.shape[:2]

        # If shapes are identical, compute once and reuse
        if seg_shape == pose_shape:
            if self.last_frame_id != frame_id or self.seg_tensor is None:
                tensor, ratio, (pad_w, pad_h) = preprocess_with_metadata(frame, seg_shape)
                metadata = {
                    "ratio": ratio,
                    "pad_w": pad_w,
                    "pad_h": pad_h,
                    "orig_h": orig_h,
                    "orig_w": orig_w
                }

                self.seg_tensor = tensor
                self.seg_metadata = metadata
                self.pose_tensor = tensor  # Reuse same tensor
                self.pose_metadata = metadata
                self.last_frame_id = frame_id

            return self.seg_tensor, self.seg_metadata, self.pose_tensor, self.pose_metadata

        # Different shapes: compute separately
        else:
            # Segmentation
            if self.last_frame_id != frame_id or self.seg_tensor is None:
                seg_tensor, seg_ratio, (seg_pad_w, seg_pad_h) = \
                    preprocess_with_metadata(frame, seg_shape)
                self.seg_tensor = seg_tensor
                self.seg_metadata = {
                    "ratio": seg_ratio,
                    "pad_w": seg_pad_w,
                    "pad_h": seg_pad_h,
                    "orig_h": orig_h,
                    "orig_w": orig_w
                }

            # Pose
            if self.last_frame_id != frame_id or self.pose_tensor is None:
                pose_tensor, pose_ratio, (pose_pad_w, pose_pad_h) = \
                    preprocess_with_metadata(frame, pose_shape)
                self.pose_tensor = pose_tensor
                self.pose_metadata = {
                    "ratio": pose_ratio,
                    "pad_w": pose_pad_w,
                    "pad_h": pose_pad_h,
                    "orig_h": orig_h,
                    "orig_w": orig_w
                }

            self.last_frame_id = frame_id

            return self.seg_tensor, self.seg_metadata, self.pose_tensor, self.pose_metadata
