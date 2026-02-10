"""
Inference engine wrapper for OpenVINO models.

This module provides a high-level interface for OpenVINO inference with
automatic device fallback (GPU → CPU) and error handling.
"""

from typing import Dict, Tuple
import numpy as np
import openvino as ov
from bakery.core.entities.model_config import ModelConfig


class InferenceEngine:
    """
    OpenVINO inference engine with device management.

    Responsibilities:
    - Compile models for target device
    - Run inference with tensors
    - Handle GPU/CPU fallback
    - Provide model metadata
    """

    def __init__(self, model_config: ModelConfig):
        """
        Initialize inference engine with model configuration.

        Args:
            model_config: Configuration containing model path and device settings

        Raises:
            FileNotFoundError: If model file does not exist
            RuntimeError: If model compilation fails on all devices
        """
        self.config = model_config
        self.core = ov.Core()
        self.compiled_model = None
        self.device = None
        self._compile()

    def _compile(self):
        """
        Compile model with device fallback (GPU → CPU).

        This method attempts to compile the model for the requested device.
        If GPU is requested but unavailable, it automatically falls back to CPU.

        Raises:
            RuntimeError: If compilation fails on all attempted devices
        """
        # Read model from file
        model = self.core.read_model(self.config.model_path)

        # Get available devices
        available_devices = self.core.available_devices

        # Attempt compilation with fallback logic
        target_device = self.config.device.value

        if target_device == "GPU" and "GPU" in available_devices:
            try:
                self.compiled_model = self.core.compile_model(model, "GPU")
                self.device = "GPU"
                return
            except RuntimeError:
                # GPU compilation failed, try CPU
                pass

        # Fallback to CPU (or if CPU was requested)
        self.compiled_model = self.core.compile_model(model, "CPU")
        self.device = "CPU"

    def infer(self, tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on input tensor.

        Args:
            tensor: Preprocessed input tensor [1, 3, H, W]

        Returns:
            Dictionary mapping output_name -> output_array

        Example:
            >>> engine = InferenceEngine(model_config)
            >>> tensor = preprocess_image(image)
            >>> outputs = engine.infer(tensor)
            >>> boxes = outputs['output0']
            >>> masks = outputs['output1']
        """
        # Run inference
        result = self.compiled_model([tensor])

        # Convert to dictionary
        outputs = {}
        for idx, output_layer in enumerate(self.compiled_model.outputs):
            # Try to get name, fallback to index-based name
            try:
                output_name = output_layer.get_any_name()
            except RuntimeError:
                # No name available, use index-based name
                output_name = f"output{idx}"

            outputs[output_name] = result[output_layer]

        return outputs

    def get_input_shape(self) -> Tuple[int, int]:
        """
        Get model input resolution (H, W).

        Returns:
            Tuple of (height, width)

        Example:
            >>> engine = InferenceEngine(model_config)
            >>> h, w = engine.get_input_shape()
            >>> print(f"Input resolution: {h}x{w}")
        """
        input_layer = self.compiled_model.input(0)
        _, _, h, w = input_layer.shape
        return (h, w)

    def get_output_shapes(self) -> Dict[str, Tuple]:
        """
        Get all output shapes.

        Returns:
            Dictionary mapping output_name -> shape_tuple

        Example:
            >>> engine = InferenceEngine(model_config)
            >>> shapes = engine.get_output_shapes()
            >>> print(f"Output 0 shape: {shapes['output0']}")
        """
        shapes = {}
        for idx, output in enumerate(self.compiled_model.outputs):
            # Try to get name, fallback to index-based name
            try:
                output_name = output.get_any_name()
            except RuntimeError:
                output_name = f"output{idx}"

            shapes[output_name] = tuple(output.shape)

        return shapes

    def get_num_outputs(self) -> int:
        """
        Get number of model outputs.

        Returns:
            Number of outputs

        Note:
            YOLO segmentation models have 2 outputs (boxes, masks)
            YOLO pose models have 1 output (keypoints + boxes)
        """
        return len(self.compiled_model.outputs)

    def get_device(self) -> str:
        """
        Get the device the model is compiled for.

        Returns:
            Device name ("GPU" or "CPU")
        """
        return self.device

    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information.

        Returns:
            Dictionary containing:
            - model_path: Path to model file
            - device: Compilation device
            - input_shape: Input tensor shape
            - output_shapes: Dictionary of output shapes
            - num_outputs: Number of outputs
        """
        return {
            "model_path": str(self.config.model_path),
            "device": self.device,
            "input_shape": self.get_input_shape(),
            "output_shapes": self.get_output_shapes(),
            "num_outputs": self.get_num_outputs()
        }
