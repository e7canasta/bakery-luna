"""
Bakery Catalog - Model Repository
===================================

Local repository for discovering and cataloging model artifacts.

ModelRepository scans the filesystem, validates file existence, extracts
metadata from directory structure, and produces ModelInfo objects.

It does NOT depend on any inference engine (OpenVINO, PyTorch, etc).
Validation is limited to file existence and directory structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bakery_catalog.constants import FormatType
from bakery_catalog.config import config
from bakery_catalog.paths import ModelPath
from bakery_catalog.model_info import ModelInfo, ModelType, Precision, ModelSize


# Mapping from format directory name → Precision enum
_FORMAT_TO_PRECISION = {
    "fp16": Precision.FP16,
    "fp32": Precision.FP32,
    "int8": Precision.INT8,
    "int8_calibrated": Precision.INT8_CALIBRATED,
}

# Mapping from task suffix → ModelType enum
_TASK_TO_MODEL_TYPE = {
    "detection": ModelType.DETECTION,
    "segmentation": ModelType.SEGMENTATION,
    "pose": ModelType.POSE,
}

# Mapping from size char → ModelSize enum
_SIZE_TO_MODEL_SIZE = {
    "n": ModelSize.NANO,
    "s": ModelSize.SMALL,
    "m": ModelSize.MEDIUM,
    "l": ModelSize.LARGE,
    "x": ModelSize.XLARGE,
}


class ModelRepository:
    """
    Local model repository — the "depósito" (warehouse).

    Scans the filesystem for model artifacts and produces ModelInfo objects.
    This class has zero dependency on any inference engine.

    Usage:
        repo = ModelRepository()                    # uses default models_dir
        repo = ModelRepository(Path("./models"))    # custom base dir

        # Discover all models
        models = repo.discover()

        # Get a specific model
        info = repo.get("yolo26n-seg", 320, "fp16")

        # Filter by task
        seg_models = repo.discover(task=ModelType.SEGMENTATION)
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the model repository.

        Args:
            base_dir: Base directory to scan. Defaults to config.models_dir.
        """
        self.base_dir = base_dir or config.models_dir

    def discover(
        self,
        task: Optional[ModelType] = None,
        yolo_version: Optional[str] = None,
    ) -> list[ModelInfo]:
        """
        Scan the filesystem and discover all valid model artifacts.

        Args:
            task: Filter by model type (segmentation, pose, detection).
            yolo_version: Filter by YOLO version ("8", "11", "26").

        Returns:
            List of ModelInfo objects for each discovered model.
        """
        models = []

        if not self.base_dir.exists():
            return models

        for model_dir in sorted(self.base_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            parsed = ModelPath.parse_model_name(model_name)

            # Apply filters
            if task and parsed["task"] != task.value:
                continue
            if yolo_version and parsed["yolo_version"] != yolo_version:
                continue

            # Scan resolution directories
            for res_dir in sorted(model_dir.iterdir()):
                if not res_dir.is_dir():
                    continue

                try:
                    resolution = int(res_dir.name)
                except ValueError:
                    continue

                # Scan format directories
                for fmt_dir in sorted(res_dir.iterdir()):
                    if not fmt_dir.is_dir():
                        continue
                    if fmt_dir.name not in _FORMAT_TO_PRECISION:
                        continue

                    info = self._build_model_info(
                        model_name, parsed, resolution, fmt_dir
                    )
                    if info is not None:
                        models.append(info)

        return models

    def get(
        self,
        model_name: str,
        resolution: int,
        format: FormatType,
    ) -> Optional[ModelInfo]:
        """
        Get a specific model by name, resolution, and format.

        Args:
            model_name: Full model name (e.g. "yolo26n-seg")
            resolution: Input resolution (e.g. 320)
            format: Model format ("fp16", "int8", etc.)

        Returns:
            ModelInfo if found, None otherwise.
        """
        model_path = ModelPath.get(model_name, resolution, format, self.base_dir)
        parsed = ModelPath.parse_model_name(model_name)

        precision = _FORMAT_TO_PRECISION.get(format)
        if precision is None:
            return None

        model_type = _TASK_TO_MODEL_TYPE.get(parsed["task"], ModelType.DETECTION)
        model_size = _SIZE_TO_MODEL_SIZE.get(parsed["size"], ModelSize.NANO)

        info = ModelInfo(
            model_path=model_path,
            model_name=model_name,
            model_type=model_type,
            resolution=resolution,
            precision=precision,
            yolo_version=parsed["yolo_version"],
            model_size=model_size,
        )

        if not info.exists():
            return None

        return info

    def validate(self, model_path: Path) -> bool:
        """
        Validate that a model file exists and has the expected companion files.

        Args:
            model_path: Path to the model file (.xml or .onnx)

        Returns:
            True if the model file (and .bin for OpenVINO) exists.
        """
        if not model_path.exists():
            return False

        if model_path.suffix == ".xml":
            return model_path.with_suffix(".bin").exists()

        return True

    def _build_model_info(
        self,
        model_name: str,
        parsed: dict,
        resolution: int,
        fmt_dir: Path,
    ) -> Optional[ModelInfo]:
        """
        Build a ModelInfo from a format directory if valid model files exist.
        """
        format_name = fmt_dir.name
        precision = _FORMAT_TO_PRECISION.get(format_name)
        if precision is None:
            return None

        model_type = _TASK_TO_MODEL_TYPE.get(parsed["task"], ModelType.DETECTION)
        model_size = _SIZE_TO_MODEL_SIZE.get(parsed["size"], ModelSize.NANO)

        # Determine model file path
        if format_name == "onnx":
            model_file = fmt_dir / "model.onnx"
        else:
            model_file = fmt_dir / "model.xml"

        if not model_file.exists():
            return None

        # For OpenVINO, also check .bin
        if model_file.suffix == ".xml":
            if not model_file.with_suffix(".bin").exists():
                return None

        return ModelInfo(
            model_path=model_file,
            model_name=model_name,
            model_type=model_type,
            resolution=resolution,
            precision=precision,
            yolo_version=parsed["yolo_version"],
            model_size=model_size,
        )


# ── Legacy compatibility functions ──────────────────────────────────────
# These wrap the new ModelRepository for backward compatibility.

_default_repo: Optional[ModelRepository] = None


def _get_default_repo() -> ModelRepository:
    """Get or create the default ModelRepository instance."""
    global _default_repo
    if _default_repo is None:
        _default_repo = ModelRepository()
    return _default_repo


def model_exists(
    model_name: str,
    resolution: int,
    format: FormatType,
    base_dir: Optional[Path] = None,
) -> bool:
    """Check if model file exists (legacy wrapper)."""
    repo = ModelRepository(base_dir) if base_dir else _get_default_repo()
    info = repo.get(model_name, resolution, format)
    return info is not None


def list_models(
    base_dir: Optional[Path] = None,
    task: Optional[str] = None,
) -> list[dict]:
    """List all available models (legacy wrapper)."""
    repo = ModelRepository(base_dir) if base_dir else _get_default_repo()
    task_filter = _TASK_TO_MODEL_TYPE.get(task) if task else None
    models = repo.discover(task=task_filter)

    # Convert to legacy dict format
    result = []
    for info in models:
        result.append({
            "name": info.model_name,
            "resolution": info.resolution,
            "formats": [info.precision.value],
            "yolo_version": info.yolo_version,
            "size": info.model_size.value,
            "task": info.model_type.value,
        })

    return result
