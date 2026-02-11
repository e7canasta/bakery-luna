"""
Bakery Catalog - Model discovery, paths, and configuration.
============================================================

Centralized configuration for the Bakery pipeline.
Provides model path building, discovery, and environment configuration.

Usage:
    from bakery_catalog import ModelRepository, ModelInfo, config

    # Get a model from the catalog
    repo = ModelRepository()
    info = repo.get("yolo26n-seg", 320, "fp16")

    # Discover all available models
    models = repo.discover()

    # Build a model path (for exporters)
    from bakery_catalog import ModelPath
    path = ModelPath.get("yolo26n-seg", 320, "fp16")
"""

from bakery_catalog.config import config, BakeryConfig
from bakery_catalog.constants import (
    FormatType,
    YOLO_VERSIONS,
    MODEL_SIZES,
    TASK_TYPES,
    RESOLUTIONS,
)
from bakery_catalog.model_info import (
    ModelInfo,
    ModelType,
    Precision,
    ModelSize,
)
from bakery_catalog.paths import ModelPath
from bakery_catalog.discovery import (
    ModelRepository,
    model_exists,
    list_models,
)

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
    # Model Info (entities)
    "ModelInfo",
    "ModelType",
    "Precision",
    "ModelSize",
    # Constants
    "FormatType",
    "YOLO_VERSIONS",
    "MODEL_SIZES",
    "TASK_TYPES",
    "RESOLUTIONS",
    # Paths
    "ModelPath",
    # Repository
    "ModelRepository",
    # Discovery (legacy)
    "model_exists",
    "list_models",
    # Convenience
    "get_model_path",
    "list_available_models",
]
