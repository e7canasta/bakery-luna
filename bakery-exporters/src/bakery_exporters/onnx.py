"""
Bakery Exporters - ONNX Export
==============================

Export YOLO models to ONNX format.

Consolidates the export_to_onnx() function from:
- export_sauron_segmentation.py
- export_sauron_pose.py
- export_int8.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bakery_catalog import config, ModelPath


class OnnxExporter:
    """
    Export YOLO models to ONNX format.

    Usage:
        exporter = OnnxExporter()
        onnx_path = exporter.export("yolo26n-seg", 320)
    """

    def __init__(self, opset: Optional[int] = None):
        """
        Initialize ONNX exporter.

        Args:
            opset: ONNX opset version (default: from config.onnx_opset)
        """
        self.opset = opset or config.onnx_opset

    def export(
        self,
        model_name: str,
        resolution: int,
        output_dir: Optional[Path] = None,
        simplify: bool = True,
        dynamic: bool = False,
    ) -> Path:
        """
        Export YOLO model to ONNX format.

        Args:
            model_name: Full model name (e.g., yolo26n-seg)
            resolution: Input resolution (e.g., 320)
            output_dir: Override output directory
            simplify: Simplify ONNX graph (default: True)
            dynamic: Use dynamic input shapes (default: False)

        Returns:
            Path to exported ONNX file

        Raises:
            ImportError: If ultralytics is not installed
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for ONNX export. "
                "Install with: pip install bakery-exporters[onnx]"
            )

        # Determine output path
        if output_dir:
            output_path = output_dir / "model.onnx"
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_path = ModelPath.build(model_name, resolution, "onnx")

        print(f"   Exporting {model_name} to ONNX ({resolution}x{resolution})...")

        # Load and export model
        model = YOLO(model_name)
        export_path = model.export(
            format="onnx",
            imgsz=resolution,
            simplify=simplify,
            dynamic=dynamic,
            opset=self.opset,
        )

        # Move to target location if needed
        onnx_file = Path(export_path)
        if onnx_file != output_path:
            if output_path.exists():
                output_path.unlink()
            onnx_file.rename(output_path)

        print(f"   ONNX exported: {output_path}")
        return output_path


def export_to_onnx(model_name: str, resolution: int) -> Path:
    """
    Convenience function for ONNX export.

    Args:
        model_name: Full model name (e.g., yolo26n-seg)
        resolution: Input resolution (e.g., 320)

    Returns:
        Path to exported ONNX file
    """
    exporter = OnnxExporter()
    return exporter.export(model_name, resolution)
