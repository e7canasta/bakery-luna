"""
Bakery Runtime - Model Instance
================================

Opaque, ready-to-use inference wrapper.

ModelInstance is what the pipeline sees. It combines:
- ModelInfo from the catalog (WHAT the model is)
- Runtime config (WHERE to run it — device, confidence)
- An engine adapter (HOW to run it — OpenVINO, ONNX, etc.)

The consumer calls .infer(tensor) and gets results.
It does not know — and does not need to know — what engine is underneath.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from bakery_catalog import ModelRepository
from bakery_catalog.model_info import ModelInfo
from bakery_catalog.constants import FormatType


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

        # From catalog directly (one-step)
        instance = ModelInstance.from_catalog(
            repo, "yolo26n-seg", 320, "fp16", device=Device.GPU
        )

        # Use it — engine-agnostic
        outputs = instance.infer(tensor)
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

    # ── Public API (engine-agnostic) ────────────────────────────────────

    def infer(self, tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on input tensor.

        Args:
            tensor: Preprocessed input tensor [1, 3, H, W]

        Returns:
            Dictionary mapping output_name → output_array
        """
        return self._engine.infer(tensor)

    def get_input_shape(self) -> Tuple[int, int]:
        """Get model input resolution (H, W)."""
        return self._engine.get_input_shape()

    def get_output_shapes(self) -> Dict[str, Tuple]:
        """Get all output shapes."""
        return self._engine.get_output_shapes()

    def get_num_outputs(self) -> int:
        """Get number of model outputs."""
        return self._engine.get_num_outputs()

    # ── Properties ──────────────────────────────────────────────────────

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
