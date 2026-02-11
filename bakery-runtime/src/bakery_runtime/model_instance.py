"""
Bakery Runtime - Model Instance
================================

Opaque, ready-to-use inference wrapper.

The "mecánico" (mechanic) — takes model artifacts from the catalog,
compiles them for the target device, and delivers ready-to-use instances.

ModelInstance is what the pipeline sees. It combines:
- ModelInfo from the catalog (WHAT the model is)
- Runtime config (WHERE to run it — device, confidence)
- An engine adapter (HOW to run it — OpenVINO, ONNX, etc.)
- Processing logic (Pre/Post processing)

The consumer calls .infer_image(frame) and gets domain results.
It does not know — and does not need to know — what engine is underneath.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple, Any, List

import numpy as np

from bakery_catalog import ModelRepository
from bakery_catalog.model_info import ModelInfo, ModelType
from bakery_catalog.constants import FormatType
from bakery_runtime.processing.image import preprocess_with_metadata
from bakery_runtime.processing.results import postprocess_segmentation, postprocess_pose


class Device(Enum):
    """Target inference device."""

    GPU = "GPU"
    CPU = "CPU"
    AUTO = "AUTO"


class ModelInstance:
    """
    Opaque model inference instance — the output of the "mecánico".

    This is the object that the pipeline works with. It abstracts away
    the inference engine completely. The consumer doesn't know if it's
    OpenVINO FP16, INT8, PyTorch, or even a remote API.

    Usage:
        # From a ModelInfo (catalog gave you the info)
        instance = ModelInstance.from_info(info, device=Device.GPU)

        # Full cycle (simplest)
        detections = instance.infer_image(frame)

        # Optimized cycle (shared preprocessing)
        tensor, metadata = instance.preprocess(frame)
        raw_outputs = instance.infer_tensor(tensor)
        detections = instance.postprocess(raw_outputs, metadata)
    """

    def __init__(
        self,
        info: ModelInfo,
        engine,  # duck-typed: must have .infer(), .get_input_shape(), etc.
        device: Device = Device.GPU,
        confidence: float = 0.25,
    ):
        """
        Initialize a model instance.

        Typically you should use the factory methods instead of __init__.

        Args:
            info: Model info from the catalog.
            engine: Compiled inference engine (OpenVINO, ONNX, etc.)
            device: Device the model is running on.
            confidence: Detection confidence threshold.
        """
        self._info = info
        self._engine = engine
        self._device = device
        self._confidence = confidence

    @classmethod
    def from_info(
        cls,
        info: ModelInfo,
        device: Device = Device.GPU,
        confidence: float = 0.25,
    ) -> "ModelInstance":
        """
        Create a ModelInstance from a ModelInfo.

        This is the primary factory method. It selects the appropriate
        engine based on the model format and compiles it.

        Args:
            info: Model info from the catalog.
            device: Target device for inference.
            confidence: Detection confidence threshold.

        Returns:
            Ready-to-use ModelInstance.
        """
        engine = cls._create_engine(info, device)
        return cls(info=info, engine=engine, device=device, confidence=confidence)

    @classmethod
    def from_catalog(
        cls,
        repo: ModelRepository,
        model_name: str,
        resolution: int,
        format: FormatType = "fp16",
        device: Device = Device.GPU,
        confidence: float = 0.25,
    ) -> "ModelInstance":
        """
        One-step factory: fetch from catalog and compile.

        Args:
            repo: Model repository (catalog).
            model_name: Full model name (e.g. "yolo26n-seg").
            resolution: Input resolution (e.g. 320).
            format: Model format ("fp16", "int8", etc.).
            device: Target device.
            confidence: Detection confidence threshold.

        Returns:
            Ready-to-use ModelInstance.

        Raises:
            FileNotFoundError: If model not found in catalog.
        """
        info = repo.get(model_name, resolution, format)
        if info is None:
            raise FileNotFoundError(
                f"Model not found in catalog: {model_name} @ {resolution}px ({format})"
            )
        return cls.from_info(info, device=device, confidence=confidence)

    @staticmethod
    def _create_engine(info: ModelInfo, device: Device):
        """
        Select and create the appropriate engine for the model.

        Currently only OpenVINO is supported. Future engines can be
        added here based on model format or configuration.
        """
        if info.is_openvino:
            from bakery_runtime.engines.openvino_engine import OpenVINOEngine
            return OpenVINOEngine(info, device=device.value)

        # Future: ONNX Runtime, PyTorch, etc.
        raise ValueError(
            f"No engine available for model format: {info.model_path.suffix}"
        )

    # ── Full Cycle API (Standard) ──────────────────────────────────────

    def infer_image(self, image: np.ndarray) -> Tuple:
        """
        Full inference cycle: Preprocess -> Infer -> Postprocess.

        Args:
            image: Input image (BGR, OpenCV format).

        Returns:
            Post-processed results (boxes, scores, class_ids, masks/keypoints).
            Format depends on model type (Segmentation or Pose).
        """
        # 1. Preprocess
        tensor, metadata = self.preprocess(image)

        # 2. Infer
        raw_outputs = self.infer_tensor(tensor)

        # 3. Postprocess
        return self.postprocess(raw_outputs, metadata)

    # ── Granular API (Optimized) ───────────────────────────────────────

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess image for inference.

        Args:
            image: Input image (BGR).

        Returns:
            Tuple of (tensor, metadata_dict).
            Metadata contains 'ratio', 'pad_w', 'pad_h', 'orig_h', 'orig_w'.
        """
        input_shape = self.get_input_shape()
        tensor, ratio, (pad_w, pad_h) = preprocess_with_metadata(image, input_shape)

        metadata = {
            "ratio": ratio,
            "pad_w": pad_w,
            "pad_h": pad_h,
            "orig_h": image.shape[0],
            "orig_w": image.shape[1],
            "input_shape": input_shape,
        }
        return tensor, metadata

    def infer_tensor(self, tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on preprocessed tensor.

        Args:
            tensor: Preprocessed input tensor [1, 3, H, W].

        Returns:
            Dictionary mapping output_name → output_array.
        """
        return self._engine.infer(tensor)

    def postprocess(self, raw_outputs: Dict[str, np.ndarray], metadata: Dict, classes: Optional[List[int]] = None) -> Tuple:
        """
        Postprocess raw model outputs into domain objects.

        Args:
            raw_outputs: Dictionary of raw output tensors.
            metadata: Metadata dictionary from preprocess().

        Returns:
            Tuple of results.
            - Segmentation: (boxes, scores, class_ids, masks)
            - Pose: (boxes, scores, class_ids, keypoints)
        """
        # Unwrap dictionary if needed (engine returns dict, postprocess expects explicit args usually)
        # But our postprocess functions expect specific arrays.
        # We need to map dict keys to expected inputs based on model type.

        # For OpenVINO YOLO models, we usually have implicit output order or names.
        # Let's rely on the engine's output values for now, assuming standard YOLO order.
        outputs_list = list(raw_outputs.values())

        if self.model_type == ModelType.SEGMENTATION:
            # YOLO seg has 2 outputs: boxes and masks
            output_boxes = outputs_list[0]
            output_masks = outputs_list[1] if len(outputs_list) > 1 else None

            if output_masks is None:
                # Fallback or error?
                raise ValueError("Segmentation model did not return masks output")

            return postprocess_segmentation(
                output_boxes,
                output_masks,
                metadata["input_shape"],
                conf_threshold=self._confidence,
                classes=classes,
            )

        elif self.model_type == ModelType.POSE:
            # YOLO pose has 1 output: keypoints+boxes
            output_data = outputs_list[0]

            return postprocess_pose(
                output_data,
                metadata["input_shape"],
                conf_threshold=self._confidence,
            )

        else:
            raise NotImplementedError(f"Postprocessing for {self.model_type} not implemented")

    # ── Legacy/Compatibility ───────────────────────────────────────────

    def infer(self, tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Legacy alias for infer_tensor.
        Kept for backward compatibility during migration.
        """
        return self.infer_tensor(tensor)

    # ── Properties & Helpers ───────────────────────────────────────────

    def get_input_shape(self) -> Tuple[int, int]:
        """Get model input resolution (H, W)."""
        return self._engine.get_input_shape()

    def get_output_shapes(self) -> Dict[str, Tuple]:
        """Get all output shapes."""
        return self._engine.get_output_shapes()

    def get_num_outputs(self) -> int:
        """Get number of model outputs."""
        return self._engine.get_num_outputs()

    @property
    def info(self) -> ModelInfo:
        """The catalog info for this model."""
        return self._info

    @property
    def device(self) -> Device:
        """The requested device."""
        return self._device

    @property
    def active_device(self) -> str:
        """The actual device the engine compiled for (may differ due to fallback)."""
        return self._engine.device

    @property
    def confidence(self) -> float:
        """Detection confidence threshold."""
        return self._confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        """Update confidence threshold at runtime."""
        if not 0 <= value <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {value}")
        self._confidence = value

    @property
    def resolution(self) -> int:
        """Input resolution (convenience shortcut)."""
        return self._info.resolution

    @property
    def model_type(self):
        """Model type (convenience shortcut)."""
        return self._info.model_type

    def get_summary(self) -> Dict:
        """Get comprehensive instance info."""
        return {
            "model_name": self._info.model_name,
            "model_path": str(self._info.model_path),
            "model_type": self._info.model_type.value,
            "resolution": self._info.resolution,
            "precision": self._info.precision.value,
            "requested_device": self._device.value,
            "active_device": self.active_device,
            "confidence": self._confidence,
            "input_shape": self.get_input_shape(),
            "num_outputs": self.get_num_outputs(),
        }
