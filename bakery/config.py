"""
Bakery Configuration Module
===========================

Centralized configuration for the Bakery pipeline.
Loads settings from environment variables and .env file.

Environment Variables:
- BAKERY_MODELS_DIR: Base directory for exported models (default: ./models)
- BAKERY_CALIBRATION_DIR: Directory for calibration data (default: ./calibration_data)
- BAKERY_DEFAULT_YOLO: Default YOLO version (default: 11)

Usage:
    from bakery.config import config, ModelPath

    # Get model directory
    model_dir = config.models_dir

    # Get path to a specific model
    path = ModelPath.get("yolo26n-seg", 320, "fp16")
    # Returns: models/yolo26n-seg/320/fp16/model.xml

    # Build a model path for export
    export_path = ModelPath.build("yolo26n-seg", 320, "int8")
    # Returns Path object, creates directories if needed
"""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Optional


# ============================================================================
# Environment Loading
# ============================================================================

def _load_dotenv() -> None:
    """Load .env file if it exists (without external dependency)."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        # Try parent directories up to 3 levels
        for _ in range(3):
            env_path = env_path.parent.parent / ".env"
            if env_path.exists():
                break
        else:
            return

    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
    except Exception:
        pass  # Silently ignore .env parsing errors


# Load .env on module import
_load_dotenv()


# ============================================================================
# Configuration Dataclass
# ============================================================================

# Supported formats
FormatType = Literal["onnx", "fp16", "int8", "int8_calibrated"]

# YOLO versions available
YOLO_VERSIONS = ["8", "11", "26"]

# Model sizes
MODEL_SIZES = ["n", "s", "m", "l", "x"]

# Task types
TASK_TYPES = ["detection", "segmentation", "pose"]

# Standard resolutions
RESOLUTIONS = [160, 192, 224, 256, 288, 320, 640]


@dataclass
class BakeryConfig:
    """
    Centralized configuration for Bakery pipeline.

    All paths are resolved relative to the project root.
    Configuration is loaded from environment variables.
    """

    # Base directories
    models_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("BAKERY_MODELS_DIR", "models")
    ))

    calibration_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("BAKERY_CALIBRATION_DIR", "calibration_data")
    ))

    # Default YOLO version
    default_yolo_version: str = field(default_factory=lambda:
        os.environ.get("BAKERY_DEFAULT_YOLO", "11")
    )

    # ONNX export settings
    onnx_opset: int = field(default_factory=lambda:
        int(os.environ.get("BAKERY_ONNX_OPSET", "17"))
    )

    def __post_init__(self):
        """Convert string paths to Path objects if needed."""
        if isinstance(self.models_dir, str):
            self.models_dir = Path(self.models_dir)
        if isinstance(self.calibration_dir, str):
            self.calibration_dir = Path(self.calibration_dir)

    def ensure_dirs(self) -> None:
        """Create base directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = BakeryConfig()


# ============================================================================
# Model Path Builder
# ============================================================================

class ModelPath:
    """
    Helper class for building consistent model paths.

    Directory Structure:
        {models_dir}/{model_name}/{resolution}/{format}/
            model.onnx (for onnx format)
            model.xml + model.bin (for OpenVINO formats)

    Examples:
        models/yolo26n-seg/320/fp16/model.xml
        models/yolo11m-pose/256/int8_calibrated/model.xml
        models/yolo26n-seg/320/onnx/model.onnx
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

    @staticmethod
    def exists(
        model_name: str,
        resolution: int,
        format: FormatType,
        base_dir: Optional[Path] = None
    ) -> bool:
        """Check if model file exists."""
        path = ModelPath.get(model_name, resolution, format, base_dir)
        if format == "onnx":
            return path.exists()
        else:
            # For OpenVINO, check both .xml and .bin
            bin_path = path.with_suffix(".bin")
            return path.exists() and bin_path.exists()

    @staticmethod
    def list_models(
        base_dir: Optional[Path] = None,
        task: Optional[str] = None
    ) -> list[dict]:
        """
        List all available models.

        Args:
            base_dir: Override base directory
            task: Filter by task type (detection, segmentation, pose)

        Returns:
            List of dicts with model info
        """
        base = base_dir or config.models_dir
        models = []

        if not base.exists():
            return models

        for model_dir in base.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            info = ModelPath.parse_model_name(model_name)

            if task and info["task"] != task:
                continue

            for res_dir in model_dir.iterdir():
                if not res_dir.is_dir():
                    continue

                try:
                    resolution = int(res_dir.name)
                except ValueError:
                    continue

                formats = []
                for fmt_dir in res_dir.iterdir():
                    if fmt_dir.is_dir() and fmt_dir.name in ["onnx", "fp16", "int8", "int8_calibrated"]:
                        formats.append(fmt_dir.name)

                if formats:
                    models.append({
                        "name": model_name,
                        "resolution": resolution,
                        "formats": formats,
                        **info
                    })

        return models


# ============================================================================
# Convenience Functions
# ============================================================================

def get_model_path(
    model_name: str,
    resolution: int,
    format: FormatType = "fp16"
) -> Path:
    """
    Convenience function to get model path.

    Args:
        model_name: Full model name (e.g., "yolo26n-seg")
        resolution: Input resolution (e.g., 320)
        format: Model format (default: fp16)

    Returns:
        Path to model file
    """
    return ModelPath.get(model_name, resolution, format)


def list_available_models(task: Optional[str] = None) -> list[dict]:
    """
    List all available exported models.

    Args:
        task: Filter by task (detection, segmentation, pose)

    Returns:
        List of model info dicts
    """
    return ModelPath.list_models(task=task)
