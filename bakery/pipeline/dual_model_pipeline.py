"""
Dual-model inference pipeline with smart scheduling.

This module orchestrates segmentation and pose estimation inference
with optimizations like preprocessing cache and smart frame scheduling.
"""

from typing import Optional, Tuple
import numpy as np
import cv2
import supervision as sv
from bakery.core.entities.frame import Frame
from bakery.core.entities.detection import Segmentation, BoundingBox, Mask
from bakery.core.entities.pose import PoseEstimation, Skeleton
from bakery.core.entities.model_config import PipelineConfig
from bakery.adapters.openvino.inference_engine import InferenceEngine
from bakery.adapters.openvino.preprocessing import PreprocessCache
from bakery.adapters.openvino.postprocessing import (
    postprocess_segmentation,
    postprocess_pose
)
from bakery.utils.metrics import PerformanceMetrics


class DualModelPipeline:
    """
    Dual-model inference pipeline with smart scheduling.

    Orchestrates:
    - Preprocessing with cache optimization
    - Segmentation inference (every N frames)
    - Pose inference (every frame)
    - Postprocessing to structured outputs
    - Performance metrics tracking
    """

    def __init__(
        self,
        seg_engine: InferenceEngine,
        pose_engine: InferenceEngine,
        config: PipelineConfig
    ):
        """
        Initialize dual-model pipeline.

        Args:
            seg_engine: Inference engine for segmentation model
            pose_engine: Inference engine for pose estimation model
            config: Pipeline configuration

        Example:
            >>> seg_engine = InferenceEngine(seg_config)
            >>> pose_engine = InferenceEngine(pose_config)
            >>> pipeline = DualModelPipeline(seg_engine, pose_engine, config)
        """
        self.seg_engine = seg_engine
        self.pose_engine = pose_engine
        self.config = config

        # Preprocessing cache
        self.preprocess_cache = PreprocessCache()

        # Metrics
        self.metrics = PerformanceMetrics()

        # Cached segmentation results
        self._cached_segmentation: Optional[Segmentation] = None

    def process_frame(
        self,
        frame: Frame
    ) -> Tuple[Segmentation, PoseEstimation]:
        """
        Process single frame through dual-model pipeline.

        Args:
            frame: Input frame

        Returns:
            Tuple of (Segmentation, PoseEstimation)

        Example:
            >>> frame = Frame.from_array(image, frame_id=0)
            >>> segmentation, poses = pipeline.process_frame(frame)
        """
        self.metrics.total_frames += 1

        # Get model input shapes
        seg_shape = self.seg_engine.get_input_shape()
        pose_shape = self.pose_engine.get_input_shape()

        # Preprocessing with cache
        seg_tensor, seg_meta, pose_tensor, pose_meta = \
            self.preprocess_cache.get_or_compute(
                frame.data, frame.frame_id,
                seg_shape, pose_shape
            )

        # Smart scheduling: Segmentation every N frames
        if frame.frame_id % self.config.seg_interval == 0:
            seg_outputs = self.seg_engine.infer(seg_tensor)

            # Get output arrays (handle different output naming)
            output_keys = list(seg_outputs.keys())
            seg_output_boxes_data = seg_outputs[output_keys[0]]  # First output: boxes
            seg_output_masks_data = seg_outputs[output_keys[1]] if len(output_keys) > 1 else None

            # Postprocess segmentation
            boxes, scores, class_ids, masks = postprocess_segmentation(
                seg_output_boxes_data,
                seg_output_masks_data,
                seg_shape,
                conf_threshold=self.config.confidence_threshold
            )

            # Create Segmentation entity
            self._cached_segmentation = self._create_segmentation(
                boxes, scores, class_ids, masks,
                frame.frame_id, seg_meta
            )
            self.metrics.seg_runs += 1

        # Reuse cached segmentation
        segmentation = self._cached_segmentation if self._cached_segmentation is not None else Segmentation.empty(frame.frame_id)

        # Always run pose
        pose_outputs = self.pose_engine.infer(pose_tensor)

        # Get output array (pose has 1 output)
        output_keys = list(pose_outputs.keys())
        pose_output_data = pose_outputs[output_keys[0]]

        # Postprocess pose
        boxes, scores, class_ids, keypoints = postprocess_pose(
            pose_output_data,
            pose_shape,
            conf_threshold=self.config.confidence_threshold
        )

        # Create PoseEstimation entity
        pose_estimation = self._create_pose_estimation(
            boxes, scores, keypoints,
            frame.frame_id, pose_meta
        )
        self.metrics.pose_runs += 1

        return segmentation, pose_estimation

    def _create_segmentation(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        masks: np.ndarray,
        frame_id: int,
        metadata: dict
    ) -> Segmentation:
        """
        Convert postprocessing outputs to Segmentation entity.

        Args:
            boxes: Detection boxes in xyxy format
            scores: Confidence scores
            class_ids: Class IDs
            masks: Binary masks
            frame_id: Frame identifier
            metadata: Preprocessing metadata (ratio, padding)

        Returns:
            Segmentation entity with transformed coordinates
        """
        if len(boxes) == 0:
            return Segmentation.empty(frame_id)

        # Apply coordinate transformation (from model space to frame space)
        ratio = metadata["ratio"]
        pad_w = metadata["pad_w"]
        pad_h = metadata["pad_h"]
        orig_h = metadata["orig_h"]
        orig_w = metadata["orig_w"]

        # Transform boxes: (x - pad) / ratio
        transformed_boxes = boxes.copy()
        transformed_boxes[:, [0, 2]] = (transformed_boxes[:, [0, 2]] - pad_w) / ratio
        transformed_boxes[:, [1, 3]] = (transformed_boxes[:, [1, 3]] - pad_h) / ratio

        # Clip boxes to frame boundaries
        transformed_boxes[:, [0, 2]] = np.clip(transformed_boxes[:, [0, 2]], 0, orig_w)
        transformed_boxes[:, [1, 3]] = np.clip(transformed_boxes[:, [1, 3]], 0, orig_h)

        # Create BoundingBox and Mask objects
        bbox_objects = []
        mask_objects = []

        for i in range(len(boxes)):
            # Create BoundingBox
            bbox = BoundingBox.from_xyxy(transformed_boxes[i], scores[i], int(class_ids[i]))
            bbox_objects.append(bbox)

            # Create Mask - resize to original frame size
            # Match original run_lens.origin.py logic exactly
            if masks is not None and len(masks) > i:
                mask = masks[i]  # Keep as float [0, 1] from sigmoid
                mask_h, mask_w = mask.shape

                # Remove letterbox padding (same logic as original)
                pad_top = int(pad_h)
                pad_bottom = int(pad_h)
                pad_left = int(pad_w)
                pad_right = int(pad_w)

                # Crop mask to remove padding
                mask_unpadded = mask[pad_top:mask_h-pad_bottom, pad_left:mask_w-pad_right]

                # Resize to original frame dimensions
                if mask_unpadded.shape[0] > 0 and mask_unpadded.shape[1] > 0:
                    mask_resized = cv2.resize(
                        mask_unpadded,
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    # Fallback: resize directly without cropping
                    mask_resized = cv2.resize(
                        mask,
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR
                    )

                # Threshold at 0.5 (mask is float [0, 1])
                mask_data = (mask_resized > 0.5).astype(np.bool_)
            else:
                # Create empty mask with original frame size
                mask_data = np.zeros((orig_h, orig_w), dtype=np.bool_)

            mask = Mask(data=mask_data, bbox=bbox)
            mask_objects.append(mask)

        return Segmentation(
            frame_id=frame_id,
            bboxes=bbox_objects,
            masks=mask_objects
        )

    def _create_pose_estimation(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        keypoints: np.ndarray,
        frame_id: int,
        metadata: dict
    ) -> PoseEstimation:
        """
        Convert postprocessing outputs to PoseEstimation entity.

        Args:
            boxes: Detection boxes in xyxy format
            scores: Confidence scores
            keypoints: Keypoints array [N, 17, 3]
            frame_id: Frame identifier
            metadata: Preprocessing metadata (ratio, padding)

        Returns:
            PoseEstimation entity with transformed coordinates
        """
        if len(boxes) == 0:
            return PoseEstimation.empty(frame_id)

        # Apply coordinate transformation
        ratio = metadata["ratio"]
        pad_w = metadata["pad_w"]
        pad_h = metadata["pad_h"]

        # Transform boxes
        transformed_boxes = boxes.copy()
        transformed_boxes[:, [0, 2]] = (transformed_boxes[:, [0, 2]] - pad_w) / ratio
        transformed_boxes[:, [1, 3]] = (transformed_boxes[:, [1, 3]] - pad_h) / ratio

        # Transform keypoints
        transformed_keypoints = keypoints.copy()
        transformed_keypoints[:, :, 0] = (transformed_keypoints[:, :, 0] - pad_w) / ratio
        transformed_keypoints[:, :, 1] = (transformed_keypoints[:, :, 1] - pad_h) / ratio

        # Create Skeleton objects
        skeleton_objects = []

        for i in range(len(boxes)):
            # Create BoundingBox for skeleton
            bbox = BoundingBox.from_xyxy(transformed_boxes[i], scores[i], class_id=0)

            # Create Skeleton
            skeleton = Skeleton.from_array(transformed_keypoints[i], bbox=bbox)
            skeleton_objects.append(skeleton)

        return PoseEstimation(
            frame_id=frame_id,
            skeletons=skeleton_objects
        )

    def get_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics.

        Returns:
            PerformanceMetrics object with pipeline statistics

        Example:
            >>> metrics = pipeline.get_metrics()
            >>> print(f"Frames processed: {metrics.total_frames}")
            >>> print(f"Segmentation runs: {metrics.seg_runs}")
        """
        return self.metrics

    def reset_metrics(self):
        """
        Reset performance metrics.

        Useful when starting a new video or processing session.
        """
        self.metrics = PerformanceMetrics()
        self._cached_segmentation = None
