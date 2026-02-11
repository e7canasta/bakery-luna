"""
Bakery Catalog Configuration
============================

Environment-based configuration for the Bakery pipeline.
Loads settings from environment variables and .env file.

Environment Variables:
- BAKERY_MODELS_DIR: Base directory for exported models (default: ./models)
- BAKERY_CALIBRATION_DIR: Directory for calibration data (default: ./calibration_data)
- BAKERY_DEFAULT_YOLO: Default YOLO version (default: 11)
- BAKERY_ONNX_OPSET: ONNX opset version (default: 17)
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field


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
