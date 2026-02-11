"""
Model repository for OpenVINO model discovery and validation.

This module provides utilities for discovering, validating, and extracting
metadata from OpenVINO YOLO models (segmentation and pose estimation).
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import openvino as ov
from bakery.core.entities.model_config import ModelConfig, ModelType, Device, Precision


class ModelRepository:
    """
    Model discovery and validation for OpenVINO models.

    Responsibilities:
    - Scan directory for .xml models
    - Extract input/output shapes
    - Validate model format (YOLO segmentation/pose)
    - Create ModelConfig entities
    """

    def __init__(self, models_dir: Path):
        """
        Initialize model repository.

        Args:
            models_dir: Directory containing .xml model files

        Raises:
            FileNotFoundError: If models_dir does not exist
        """
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        self.models_dir = models_dir
        self._core = ov.Core()

    def discover_models(self, yolo_version: Optional[str] = None) -> List[ModelConfig]:
        """
        Scan directory and return valid ModelConfig objects.

        Args:
            yolo_version: Optional YOLO version filter (e.g., "26", "11", "8").
                         Filters models by name/path containing "yolo{version}".

        Returns:
            List of valid ModelConfig objects discovered in the directory

        Note:
            Invalid models are silently filtered out. Use validate_model()
            to check why a specific model is invalid.
        """
        models = []

        # Recursively search for .xml files
        for xml_path in self.models_dir.rglob("*.xml"):
            # Filter by YOLO version if specified
            if yolo_version:
                path_str = str(xml_path).lower()
                # Check if path contains yolo{version} pattern
                # e.g., "yolo26" in path or "yolo11" in path
                yolo_pattern = f"yolo{yolo_version}"
                if yolo_pattern not in path_str:
                    continue

            if self.validate_model(xml_path):
                try:
                    metadata = self.extract_metadata(xml_path)
                    model_config = self._create_model_config(xml_path, metadata)
                    models.append(model_config)
                except Exception:
                    # Skip models that can't be parsed
                    continue

        return models

    def get_model_by_type(self, model_type: ModelType, yolo_version: Optional[str] = None) -> Optional[ModelConfig]:
        """
        Find first model matching type (segmentation/pose).

        Args:
            model_type: Type of model to search for
            yolo_version: Optional YOLO version filter (e.g., "26", "11", "8")

        Returns:
            First ModelConfig matching the type, or None if not found
        """
        for model in self.discover_models(yolo_version=yolo_version):
            if model.model_type == model_type:
                return model
        return None

    def validate_model(self, model_path: Path) -> bool:
        """
        Check if model is valid YOLO format.

        Args:
            model_path: Path to .xml model file

        Returns:
            True if model is valid (1 or 2 outputs), False otherwise

        Note:
            Valid YOLO models have:
            - Segmentation: 2 outputs (boxes, masks)
            - Pose: 1 output (keypoints + boxes)
        """
        if not model_path.exists() or model_path.suffix != ".xml":
            return False

        try:
            model = self._core.read_model(model_path)
            num_outputs = len(model.outputs)

            # Valid YOLO models have 1 (pose) or 2 (segmentation) outputs
            return num_outputs in (1, 2)

        except Exception:
            return False

    def extract_metadata(self, model_path: Path) -> Dict:
        """
        Extract input/output shapes, precision, etc.

        Args:
            model_path: Path to .xml model file

        Returns:
            Dictionary containing:
            - input_shape: Tuple (batch, channels, height, width)
            - output_shapes: List of tuples for each output
            - num_outputs: Number of model outputs

        Raises:
            RuntimeError: If model cannot be read
        """
        model = self._core.read_model(model_path)

        # Extract input shape
        input_layer = model.input(0)
        input_shape = tuple(input_layer.shape)

        # Extract output shapes
        output_shapes = [tuple(output.shape) for output in model.outputs]

        return {
            "input_shape": input_shape,
            "output_shapes": output_shapes,
            "num_outputs": len(model.outputs)
        }

    def _create_model_config(self, model_path: Path, metadata: Dict) -> ModelConfig:
        """
        Create ModelConfig from model path and metadata.

        Args:
            model_path: Path to model file
            metadata: Metadata extracted from model

        Returns:
            ModelConfig instance
        """
        # Determine model type based on filename and number of outputs
        # Priority: filename pattern > output count
        filename = model_path.stem.lower()
        num_outputs = metadata["num_outputs"]

        if "-seg" in filename or "seg_" in filename:
            model_type = ModelType.SEGMENTATION
        elif "-pose" in filename or "pose_" in filename:
            model_type = ModelType.POSE
        elif num_outputs == 2:
            # Fallback: 2 outputs typically means segmentation
            model_type = ModelType.SEGMENTATION
        else:
            # Skip models that are not segmentation or pose
            # (detection, classification, OBB, etc.)
            raise ValueError(f"Model {filename} is not segmentation or pose")

        # Extract resolution from input shape [1, 3, H, W]
        _, _, h, w = metadata["input_shape"]
        if h != w:
            raise ValueError(f"Non-square input resolution: {h}x{w}")

        resolution = h

        # Detect precision from model path (int8 vs fp16 vs fp32)
        path_str = str(model_path).lower()
        if "int8" in path_str:
            precision = Precision.INT8
        elif "fp16" in path_str:
            precision = Precision.FP16
        else:
            precision = Precision.FP32

        # Default device
        device = Device.GPU

        # Default confidence
        confidence = 0.25

        return ModelConfig(
            model_path=model_path,
            model_type=model_type,
            resolution=resolution,
            device=device,
            precision=precision,
            confidence=confidence
        )
