"""
Bakery Exporters Configuration

Configuration management for standalone bakery-exporters usage.

Can be used in two modes:
1. Embedded: Within bakery-luna (uses bakery.catalog.config)
2. Standalone: Independent package (uses this ExportersConfig)

Environment Variables:
- BAKERY_MODELS_DIR: Base directory for exported models (default: ./models)
- BAKERY_CALIBRATION_DIR: Directory for calibration data (default: ./calibration_data)
- BAKERY_ONNX_OPSET: ONNX opset version (default: 17)
- BAKERY_DEFAULT_YOLO: Default YOLO version (default: 11)
- BAKERY_EXPORTERS_CACHE_DIR: Cache directory for exporters (default: ./.cache/exporters)
- BAKERY_EXPORTERS_TEMP_DIR: Temporary directory for exports (default: $TEMP/.bakery)
- BAKERY_EXPORTERS_LOG_LEVEL: Logging level (default: INFO)
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


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
class ExportersConfig:
    """
    Configuration for Bakery Exporters.

    Can be used standalone or embedded within bakery-luna.
    All paths are resolved relative to the project root.
    """

    # ========================================================================
    # Core Paths (shared with catalog)
    # ========================================================================

    models_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("BAKERY_MODELS_DIR", "models")
    ))
    """Base directory for exported models"""

    calibration_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("BAKERY_CALIBRATION_DIR", "calibration_data")
    ))
    """Directory for calibration frames (.npy files)"""

    # ========================================================================
    # Exporter-Specific Paths
    # ========================================================================

    cache_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("BAKERY_EXPORTERS_CACHE_DIR", ".cache/exporters")
    ))
    """Cache directory for intermediate exports (ONNX files during conversion)"""

    temp_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("BAKERY_EXPORTERS_TEMP_DIR", ".tmp/bakery")
    ))
    """Temporary directory for export operations"""

    # ========================================================================
    # ONNX Settings
    # ========================================================================

    onnx_opset: int = field(default_factory=lambda:
        int(os.environ.get("BAKERY_ONNX_OPSET", "17"))
    )
    """ONNX opset version for OpenVINO compatibility"""

    onnx_simplify: bool = field(default_factory=lambda:
        os.environ.get("BAKERY_ONNX_SIMPLIFY", "true").lower() in ("true", "1", "yes")
    )
    """Simplify ONNX graph during export"""

    # ========================================================================
    # Model Settings
    # ========================================================================

    default_yolo_version: str = field(default_factory=lambda:
        os.environ.get("BAKERY_DEFAULT_YOLO", "11")
    )
    """Default YOLO version"""

    # ========================================================================
    # INT8 Quantization Settings
    # ========================================================================

    int8_synthetic_samples: int = field(default_factory=lambda:
        int(os.environ.get("BAKERY_INT8_SYNTHETIC_SAMPLES", "100"))
    )
    """Number of synthetic calibration samples for INT8"""

    int8_preset: str = field(default_factory=lambda:
        os.environ.get("BAKERY_INT8_PRESET", "MIXED")
    )
    """NNCF quantization preset (PERFORMANCE or MIXED)"""

    # ========================================================================
    # Logging Settings
    # ========================================================================

    log_level: str = field(default_factory=lambda:
        os.environ.get("BAKERY_EXPORTERS_LOG_LEVEL", "INFO")
    )
    """Logging level (DEBUG, INFO, WARNING, ERROR)"""

    verbose: bool = field(default_factory=lambda:
        os.environ.get("BAKERY_EXPORTERS_VERBOSE", "false").lower() in ("true", "1", "yes")
    )
    """Enable verbose output"""

    def __post_init__(self):
        """Convert string paths to Path objects if needed."""
        if isinstance(self.models_dir, str):
            self.models_dir = Path(self.models_dir)
        if isinstance(self.calibration_dir, str):
            self.calibration_dir = Path(self.calibration_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if isinstance(self.temp_dir, str):
            self.temp_dir = Path(self.temp_dir)

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_catalog(cls, catalog_config) -> ExportersConfig:
        """
        Create ExportersConfig from bakery.catalog.BakeryConfig.

        Used when bakery-exporters is embedded in bakery-luna.

        Args:
            catalog_config: BakeryConfig instance from bakery.catalog

        Returns:
            ExportersConfig with values from catalog_config
        """
        return cls(
            models_dir=catalog_config.models_dir,
            calibration_dir=catalog_config.calibration_dir,
            onnx_opset=catalog_config.onnx_opset,
            default_yolo_version=catalog_config.default_yolo_version,
        )

    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        return {
            "models_dir": str(self.models_dir),
            "calibration_dir": str(self.calibration_dir),
            "cache_dir": str(self.cache_dir),
            "temp_dir": str(self.temp_dir),
            "onnx_opset": self.onnx_opset,
            "onnx_simplify": self.onnx_simplify,
            "default_yolo_version": self.default_yolo_version,
            "int8_synthetic_samples": self.int8_synthetic_samples,
            "int8_preset": self.int8_preset,
            "log_level": self.log_level,
            "verbose": self.verbose,
        }


# Global config instance (standalone mode)
config = ExportersConfig()
