"""
Bakery - Modular Vision Pipeline Package
=========================================

A production-ready computer vision pipeline for dual-model inference
(segmentation + pose estimation) optimized for OpenVINO runtime.

Architecture:
- catalog: Model discovery, paths, and configuration
- core: Domain entities and business logic
- adapters: External integrations (OpenVINO, Supervision, CLI)
- utils: Shared utilities (geometry, metrics)

Project Phases:
- Luna: Refactorization (modular package)
- Juno: Single-stream production pipeline
- Neon: Multi-stream analytics platform (Metropolis-style)

Usage:
    from bakery.catalog import config, ModelPath

    # Get model path
    path = ModelPath.get("yolo26n-seg", 320, "fp16")
"""

# Import from catalog (canonical location)
from bakery.catalog import config, ModelPath, get_model_path, list_available_models

__version__ = "0.1.0"
__project_phase__ = "Luna"

__all__ = [
    "config",
    "ModelPath",
    "get_model_path",
    "list_available_models",
]
