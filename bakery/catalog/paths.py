"""
Bakery Catalog Paths
====================

Model path builder for consistent directory structure.

Directory Structure:
    {models_dir}/{model_name}/{resolution}/{format}/
        model.onnx (for onnx format)
        model.xml + model.bin (for OpenVINO formats)

Examples:
    models/yolo26n-seg/320/fp16/model.xml
    models/yolo11m-pose/256/int8_calibrated/model.xml
    models/yolo26n-seg/320/onnx/model.onnx
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bakery.catalog.constants import FormatType
from bakery.catalog.config import config


class ModelPath:
    """
    Helper class for building consistent model paths.
    """

    @staticmethod
    def get_model_name(
        yolo_version: str,
        size: str,
        task: str
    ) -> str:
        """
        Build model name from components.

        Args:
            yolo_version: YOLO version (8, 11, 26)
            size: Model size (n, s, m, l, x)
            task: Task type (detection, segmentation, pose)

        Returns:
            Model name string (e.g., "yolo26n-seg")
        """
        suffix_map = {
            "detection": "",
            "segmentation": "-seg",
            "pose": "-pose",
        }
        suffix = suffix_map.get(task, "")
        return f"yolo{yolo_version}{size}{suffix}"

    @staticmethod
    def parse_model_name(model_name: str) -> dict:
        """
        Parse model name into components.

        Args:
            model_name: Full model name (e.g., "yolo26n-seg")

        Returns:
            Dict with yolo_version, size, task
        """
        # Remove "yolo" prefix
        name = model_name.lower()
        if name.startswith("yolo"):
            name = name[4:]

        # Extract version (digits at start)
        version = ""
        idx = 0
        while idx < len(name) and name[idx].isdigit():
            version += name[idx]
            idx += 1

        # Extract size (single letter)
        size = name[idx] if idx < len(name) else "n"
        idx += 1

        # Extract task from suffix
        suffix = name[idx:] if idx < len(name) else ""
        task_map = {
            "-seg": "segmentation",
            "-pose": "pose",
            "": "detection",
        }
        task = task_map.get(suffix, "detection")

        return {
            "yolo_version": version or "11",
            "size": size,
            "task": task,
        }

    @staticmethod
    def get_dir(
        model_name: str,
        resolution: int,
        format: FormatType,
        base_dir: Optional[Path] = None
    ) -> Path:
        """
        Get directory path for a model configuration.

        Args:
            model_name: Full model name (e.g., "yolo26n-seg")
            resolution: Input resolution (e.g., 320)
            format: Model format (onnx, fp16, int8, int8_calibrated)
            base_dir: Override base directory (default: config.models_dir)

        Returns:
            Path to model directory
        """
        base = base_dir or config.models_dir
        return base / model_name / str(resolution) / format

    @staticmethod
    def get(
        model_name: str,
        resolution: int,
        format: FormatType,
        base_dir: Optional[Path] = None
    ) -> Path:
        """
        Get full path to model file.

        Args:
            model_name: Full model name (e.g., "yolo26n-seg")
            resolution: Input resolution (e.g., 320)
            format: Model format (onnx, fp16, int8, int8_calibrated)
            base_dir: Override base directory

        Returns:
            Path to model file (.xml for OpenVINO, .onnx for ONNX)
        """
        dir_path = ModelPath.get_dir(model_name, resolution, format, base_dir)

        if format == "onnx":
            return dir_path / "model.onnx"
        else:
            return dir_path / "model.xml"

    @staticmethod
    def build(
        model_name: str,
        resolution: int,
        format: FormatType,
        base_dir: Optional[Path] = None,
        create_dir: bool = True
    ) -> Path:
        """
        Build model path and optionally create directories.

        Args:
            model_name: Full model name (e.g., "yolo26n-seg")
            resolution: Input resolution (e.g., 320)
            format: Model format (onnx, fp16, int8, int8_calibrated)
            base_dir: Override base directory
            create_dir: Create directory if it doesn't exist

        Returns:
            Path to model file
        """
        dir_path = ModelPath.get_dir(model_name, resolution, format, base_dir)

        if create_dir:
            dir_path.mkdir(parents=True, exist_ok=True)

        return ModelPath.get(model_name, resolution, format, base_dir)
