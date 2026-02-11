"""
Bakery Catalog Discovery
========================

Model discovery utilities for finding and listing exported models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bakery.catalog.constants import FormatType
from bakery.catalog.config import config
from bakery.catalog.paths import ModelPath


def model_exists(
    model_name: str,
    resolution: int,
    format: FormatType,
    base_dir: Optional[Path] = None
) -> bool:
    """
    Check if model file exists.

    Args:
        model_name: Full model name (e.g., "yolo26n-seg")
        resolution: Input resolution (e.g., 320)
        format: Model format (onnx, fp16, int8, int8_calibrated)
        base_dir: Override base directory

    Returns:
        True if model exists
    """
    path = ModelPath.get(model_name, resolution, format, base_dir)
    if format == "onnx":
        return path.exists()
    else:
        # For OpenVINO, check both .xml and .bin
        bin_path = path.with_suffix(".bin")
        return path.exists() and bin_path.exists()


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
        List of dicts with model info:
        - name: Model name
        - resolution: Input resolution
        - formats: Available formats
        - yolo_version: YOLO version
        - size: Model size
        - task: Task type
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
