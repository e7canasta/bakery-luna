"""
Bakery Configuration Module (DEPRECATED)
=========================================

.. deprecated::
    This module is deprecated. Use ``bakery.catalog`` instead.

    Old import:
        from bakery.config import config, ModelPath

    New import:
        from bakery_catalog import config, ModelPath

This module re-exports from ``bakery.catalog`` for backwards compatibility.
A deprecation warning is emitted on import.
"""

import warnings

warnings.warn(
    "bakery.config is deprecated. Use bakery.catalog instead. "
    "Example: from bakery_catalog import config, ModelPath",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from catalog for backwards compatibility
from bakery_catalog import (
    # Config
    config,
    BakeryConfig,
    # Constants
    FormatType,
    YOLO_VERSIONS,
    MODEL_SIZES,
    TASK_TYPES,
    RESOLUTIONS,
    # Paths
    ModelPath,
    # Discovery
    model_exists,
    list_models,
    # Convenience
    get_model_path,
    list_available_models,
)

__all__ = [
    "config",
    "BakeryConfig",
    "FormatType",
    "YOLO_VERSIONS",
    "MODEL_SIZES",
    "TASK_TYPES",
    "RESOLUTIONS",
    "ModelPath",
    "model_exists",
    "list_models",
    "get_model_path",
    "list_available_models",
]
