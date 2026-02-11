"""
Bakery Runtime - OpenVINO Inference Engine
==========================================

OpenVINO-specific adapter for model compilation and inference.
This module knows about OpenVINO internals. Nothing else in bakery-runtime
or bakery-luna should import openvino directly.

Responsibilities:
- Read model files (.xml/.bin)
- Compile model for target device (GPU/CPU)
- Handle automatic device fallback (GPU → CPU)
- Execute inference on tensors
- Report model metadata (input/output shapes)
"""

from __future__ import annotations

from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import openvino as ov

from bakery_catalog.model_info import ModelInfo


class OpenVINOEngine:
    """
    OpenVINO inference engine adapter.

    Compiles a model for a target device and provides a simple
    infer(tensor) → dict interface. Handles device fallback automatically.
    """

    def __init__(self, model_info: ModelInfo, device: str = "GPU"):
        """
        Compile model for the target device.

        Args:
            model_info: Model artifact info from the catalog.
            device: Target device ("GPU", "CPU", "AUTO").

        Raises:
            FileNotFoundError: If model file does not exist.
            RuntimeError: If compilation fails on all devices.
        """
        if not model_info.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_info.model_path}"
            )

        self._model_info = model_info
        self._core = ov.Core()
        self._compiled_model = None
        self._active_device: str = ""

        self._compile(device)

    def _compile(self, target_device: str) -> None:
        """
        Compile model with device fallback (GPU → CPU).

        Raises:
            RuntimeError: If compilation fails on all attempted devices.
        """
        model = self._core.read_model(str(self._model_info.model_path))
        available_devices = self._core.available_devices

        # Try requested device first
        if target_device == "GPU" and "GPU" in available_devices:
            try:
                self._compiled_model = self._core.compile_model(model, "GPU")
                self._active_device = "GPU"
                return
            except RuntimeError:
                pass  # Fall through to CPU

        # Fallback to CPU
        self._compiled_model = self._core.compile_model(model, "CPU")
        self._active_device = "CPU"

    def infer(self, tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on input tensor.

        Args:
            tensor: Preprocessed input tensor [1, 3, H, W]

        Returns:
            Dictionary mapping output_name → output_array
        """
        result = self._compiled_model([tensor])

        outputs = {}
        for idx, output_layer in enumerate(self._compiled_model.outputs):
            try:
                output_name = output_layer.get_any_name()
            except RuntimeError:
                output_name = f"output{idx}"
            outputs[output_name] = result[output_layer]

        return outputs

    def get_input_shape(self) -> Tuple[int, int]:
        """Get model input resolution (H, W)."""
        input_layer = self._compiled_model.input(0)
        _, _, h, w = input_layer.shape
        return (h, w)

    def get_output_shapes(self) -> Dict[str, Tuple]:
        """Get all output shapes."""
        shapes = {}
        for idx, output in enumerate(self._compiled_model.outputs):
            try:
                name = output.get_any_name()
            except RuntimeError:
                name = f"output{idx}"
            shapes[name] = tuple(output.shape)
        return shapes

    def get_num_outputs(self) -> int:
        """Get number of model outputs."""
        return len(self._compiled_model.outputs)

    @property
    def device(self) -> str:
        """The device the model is actually compiled for."""
        return self._active_device

    @property
    def model_info(self) -> ModelInfo:
        """The catalog info used to build this engine."""
        return self._model_info
