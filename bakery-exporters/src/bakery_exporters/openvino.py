"""
Bakery Exporters - OpenVINO Conversion
======================================

Convert ONNX models to OpenVINO IR format.

Consolidates conversion logic from:
- export_sauron_segmentation.py (FP16)
- export_sauron_pose.py (FP16)
- export_int8.py (INT8 synthetic)
- calibrate_int8.py (INT8 calibrated)

Supports:
- FP16 compression for GPU inference
- INT8 quantization with synthetic data
- INT8 calibration with real data
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterator
from enum import Enum

from bakery.catalog import config, ModelPath


class OpenVINOPrecision(Enum):
    """OpenVINO model precision types."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT8_CALIBRATED = "int8_calibrated"


class OpenVINOConverter:
    """
    Convert ONNX models to OpenVINO IR format.

    Usage:
        converter = OpenVINOConverter()

        # FP16 for GPU
        fp16_path = converter.to_fp16(onnx_path, "yolo26n-seg", 320)

        # INT8 with synthetic data (fast)
        int8_path = converter.to_int8_synthetic(onnx_path, "yolo26n-seg", 320)

        # INT8 with real calibration data (accurate)
        int8_cal_path = converter.to_int8_calibrated(
            onnx_path, "yolo26n-seg", 320, calibration_data
        )
    """

    def __init__(self):
        """Initialize converter (lazy loads OpenVINO)."""
        self._core = None

    @property
    def core(self):
        """Lazy-load OpenVINO core."""
        if self._core is None:
            try:
                import openvino as ov
                self._core = ov.Core()
            except ImportError:
                raise ImportError(
                    "openvino is required for IR conversion. "
                    "Install with: pip install bakery-exporters[openvino]"
                )
        return self._core

    def to_fp16(
        self,
        onnx_path: Path,
        model_name: str,
        resolution: int,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Convert ONNX to FP16 OpenVINO IR.

        Optimized for GPU inference with reduced memory footprint.

        Args:
            onnx_path: Path to ONNX file
            model_name: Model name for output path
            resolution: Resolution for output path
            output_dir: Override output directory

        Returns:
            Path to OpenVINO model (.xml)

        Raises:
            FileNotFoundError: If ONNX file doesn't exist
        """
        import openvino as ov

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")

        # Determine output path
        if output_dir:
            output_path = output_dir / "model.xml"
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_path = ModelPath.build(model_name, resolution, "fp16")

        print(f"   Converting to FP16...")

        model = self.core.read_model(onnx_path)
        ov.save_model(model, output_path, compress_to_fp16=True)

        print(f"   FP16 exported: {output_path}")
        return output_path

    def to_int8_synthetic(
        self,
        onnx_path: Path,
        model_name: str,
        resolution: int,
        num_samples: int = 100,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Quantize to INT8 using synthetic calibration data.

        Fast quantization using random data. Good for testing,
        but calibrate_int8 with real data gives better accuracy.

        Args:
            onnx_path: Path to ONNX file
            model_name: Model name for output path
            resolution: Resolution for calibration
            num_samples: Number of synthetic samples (default: 100)
            output_dir: Override output directory

        Returns:
            Path to OpenVINO INT8 model (.xml)

        Raises:
            FileNotFoundError: If ONNX file doesn't exist
            ImportError: If nncf is not installed
        """
        import openvino as ov
        import numpy as np

        try:
            import nncf
        except ImportError:
            raise ImportError(
                "nncf is required for INT8 quantization. "
                "Install with: pip install bakery-exporters[nncf]"
            )

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")

        # Determine output path
        if output_dir:
            output_path = output_dir / "model.xml"
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_path = ModelPath.build(model_name, resolution, "int8")

        print(f"   Quantizing to INT8 (synthetic data)...")

        model = self.core.read_model(str(onnx_path))

        # Generate synthetic calibration data
        print(f"   Generating {num_samples} synthetic samples...")
        synthetic_samples = [
            np.random.rand(1, 3, resolution, resolution).astype(np.float32)
            for _ in range(num_samples)
        ]

        print(f"   Applying INT8 quantization with NNCF...")
        quantized_model = nncf.quantize(
            model,
            nncf.Dataset(synthetic_samples),
            preset=nncf.QuantizationPreset.PERFORMANCE,
            subset_size=num_samples,
        )

        ov.save_model(quantized_model, str(output_path))

        print(f"   INT8 exported: {output_path}")
        print(f"   TIP: For better accuracy, use to_int8_calibrated() with real data")
        return output_path

    def to_int8_calibrated(
        self,
        onnx_path: Path,
        model_name: str,
        resolution: int,
        calibration_data: Iterator,
        preset: str = "MIXED",
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Quantize to INT8 using real calibration data.

        Higher accuracy than synthetic quantization.

        Args:
            onnx_path: Path to ONNX file
            model_name: Model name for output path
            resolution: Resolution for calibration
            calibration_data: Iterator yielding numpy arrays (use CalibrationDataLoader)
            preset: NNCF preset - "PERFORMANCE" or "MIXED" (default: MIXED)
            output_dir: Override output directory

        Returns:
            Path to OpenVINO INT8 calibrated model (.xml)

        Raises:
            FileNotFoundError: If ONNX file doesn't exist
            ImportError: If nncf is not installed
        """
        import openvino as ov

        try:
            import nncf
        except ImportError:
            raise ImportError(
                "nncf is required for INT8 calibration. "
                "Install with: pip install bakery-exporters[nncf]"
            )

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")

        # Determine output path
        if output_dir:
            output_path = output_dir / "model.xml"
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_path = ModelPath.build(model_name, resolution, "int8_calibrated")

        print(f"   Quantizing to INT8 (calibrated)...")

        model = self.core.read_model(str(onnx_path))

        # Select preset
        preset_enum = (
            nncf.QuantizationPreset.PERFORMANCE
            if preset == "PERFORMANCE"
            else nncf.QuantizationPreset.MIXED
        )

        print(f"   Applying INT8 quantization with NNCF (preset: {preset})...")
        quantized_model = nncf.quantize(
            model,
            nncf.Dataset(calibration_data),
            preset=preset_enum,
        )

        ov.save_model(quantized_model, str(output_path))

        print(f"   INT8 calibrated exported: {output_path}")
        return output_path


# Convenience functions for backwards compatibility
def convert_to_fp16(onnx_path: Path, model_name: str, resolution: int) -> Path:
    """Convert ONNX to FP16 OpenVINO IR."""
    converter = OpenVINOConverter()
    return converter.to_fp16(onnx_path, model_name, resolution)


def quantize_to_int8(onnx_path: Path, model_name: str, resolution: int) -> Path:
    """Quantize ONNX to INT8 with synthetic data."""
    converter = OpenVINOConverter()
    return converter.to_int8_synthetic(onnx_path, model_name, resolution)
