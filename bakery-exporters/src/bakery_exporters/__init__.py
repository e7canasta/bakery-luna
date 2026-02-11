"""
Bakery Exporters - Export and conversion pipelines for YOLO models.
====================================================================

A modular package for exporting YOLO models to various formats
optimized for different hardware targets.

Can be used in two modes:
1. Embedded: Within bakery-luna (uses bakery.catalog config)
2. Standalone: Independent package (uses bakery_exporters.config)

Main Components:
- OnnxExporter: Export YOLO models to ONNX format
- OpenVINOConverter: Convert ONNX to OpenVINO IR (FP16, INT8)
- CalibrationDataLoader: Load calibration data for INT8 quantization
- ExportPipeline: Orchestrate the complete export workflow
- ExportFormat: Enum of supported export formats
- ExportersConfig: Standalone configuration management

Usage (Embedded in bakery-luna):
    from bakery_exporters import ExportPipeline, ExportFormat
    from bakery_catalog import config  # Uses catalog config

    pipeline = ExportPipeline()
    results = pipeline.export(...)

Usage (Standalone):
    from bakery_exporters import ExportPipeline, ExportFormat, config

    # config is ExportersConfig instance with env vars support
    config.models_dir = Path("/custom/models")

    pipeline = ExportPipeline()
    results = pipeline.export(...)
"""

from bakery_exporters.config import ExportersConfig, config
from bakery_exporters.onnx import OnnxExporter
from bakery_exporters.openvino import OpenVINOConverter, OpenVINOPrecision
from bakery_exporters.calibration import CalibrationDataLoader
from bakery_exporters.pipeline import ExportPipeline, ExportFormat, ExportResult

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "config",
    "ExportersConfig",
    # Core exporters
    "OnnxExporter",
    "OpenVINOConverter",
    "OpenVINOPrecision",
    "CalibrationDataLoader",
    # Pipeline
    "ExportPipeline",
    "ExportFormat",
    "ExportResult",
]
