"""
Bakery Catalog - Model discovery, paths, and configuration.
============================================================

Centralized configuration for the Bakery pipeline.
Provides model path building, discovery, and environment configuration.

Usage:
    from bakery.catalog import config, ModelPath

    # Get model directory
    model_dir = config.models_dir

    # Get path to a specific model
    path = ModelPath.get("yolo26n-seg", 320, "fp16")
    # Returns: models/yolo26n-seg/320/fp16/model.xml
"""

from bakery.catalog.config import config, BakeryConfig
from bakery.catalog.constants import (
    FormatType,
    YOLO_VERSIONS,
    MODEL_SIZES,
    TASK_TYPES,
    RESOLUTIONS,
)
from bakery.catalog.paths import ModelPath
from bakery.catalog.discovery import model_exists, list_models

# Convenience functions
def get_model_path(model_name: str, resolution: int, format: FormatType = "fp16"):
    """Convenience function to get model path."""
    return ModelPath.get(model_name, resolution, format)


def list_available_models(task: str = None) -> list[dict]:
    """List all available exported models."""
    return list_models(task=task)


__all__ = [
    # Config
    "config",
    "BakeryConfig",
    # Constants
    "FormatType",
    "YOLO_VERSIONS",
    "MODEL_SIZES",
    "TASK_TYPES",
    "RESOLUTIONS",
    # Paths
    "ModelPath",
    # Discovery
    "model_exists",
    "list_models",
    # Convenience
    "get_model_path",
    "list_available_models",
]
