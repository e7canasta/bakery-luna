"""
Bakery Exporters - Export Pipeline
==================================

Orchestrates the complete model export workflow.

Consolidates the export_model() functions from:
- export_sauron_segmentation.py
- export_sauron_pose.py
- export_int8.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from bakery.catalog import config, ModelPath

from bakery_exporters.onnx import OnnxExporter
from bakery_exporters.openvino import OpenVINOConverter
from bakery_exporters.calibration import CalibrationDataLoader


class ExportFormat(Enum):
    """Available export formats."""
    ONNX = "onnx"
    FP16 = "fp16"
    INT8 = "int8"
    INT8_CALIBRATED = "int8_calibrated"


@dataclass
class ExportResult:
    """Result of an export operation."""
    model_name: str
    resolution: int
    format: ExportFormat
    output_path: Path
    success: bool
    error: Optional[str] = None


class ExportPipeline:
    """
    Orchestrates model export workflow.

    Handles the complete pipeline:
    1. Export to ONNX (always required as intermediate)
    2. Convert to requested formats (FP16, INT8)

    Usage:
        pipeline = ExportPipeline()
        results = pipeline.export(
            yolo_version="26",
            size="n",
            task="segmentation",
            resolution=320,
            formats=[ExportFormat.FP16, ExportFormat.INT8]
        )
    """

    def __init__(self):
        """Initialize export pipeline."""
        self.onnx_exporter = OnnxExporter()
        self.openvino_converter = OpenVINOConverter()

    def export(
        self,
        yolo_version: str,
        size: str,
        task: str,
        resolution: int,
        formats: List[ExportFormat],
        dry_run: bool = False,
    ) -> List[ExportResult]:
        """
        Export a model to multiple formats.

        Args:
            yolo_version: YOLO version (8, 11, 26)
            size: Model size (n, s, m, l, x)
            task: Task type (detection, segmentation, pose)
            resolution: Input resolution
            formats: List of formats to export
            dry_run: If True, only show what would be exported

        Returns:
            List of ExportResult objects
        """
        model_name = ModelPath.get_model_name(yolo_version, size, task)
        results = []

        print(f"\n{'='*70}")
        print(f"Exporting: {model_name} @ {resolution}x{resolution}")
        print(f"{'='*70}")
        print(f"   Model: {model_name}")
        print(f"   Task: {task}")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Formats: {', '.join(f.value for f in formats)}")
        print(f"   Destination: {config.models_dir}/{model_name}/{resolution}/")

        if dry_run:
            print(f"   DRY-RUN: No files will be created")
            return results

        try:
            # Step 1: Export to ONNX (always needed as intermediate)
            print(f"\nStep 1: Exporting to ONNX...")
            onnx_path = self.onnx_exporter.export(model_name, resolution)

            if ExportFormat.ONNX in formats:
                results.append(ExportResult(
                    model_name, resolution, ExportFormat.ONNX,
                    onnx_path, True
                ))

            # Step 2: Convert to FP16 if requested
            if ExportFormat.FP16 in formats:
                print(f"\nStep 2: Converting to FP16...")
                fp16_path = self.openvino_converter.to_fp16(
                    onnx_path, model_name, resolution
                )
                results.append(ExportResult(
                    model_name, resolution, ExportFormat.FP16,
                    fp16_path, True
                ))

            # Step 3: Quantize to INT8 (synthetic) if requested
            if ExportFormat.INT8 in formats:
                print(f"\nStep 3: Quantizing to INT8...")
                int8_path = self.openvino_converter.to_int8_synthetic(
                    onnx_path, model_name, resolution
                )
                results.append(ExportResult(
                    model_name, resolution, ExportFormat.INT8,
                    int8_path, True
                ))

            # Step 4: Calibrate INT8 if requested
            if ExportFormat.INT8_CALIBRATED in formats:
                print(f"\nStep 4: Calibrating INT8 with real data...")
                try:
                    calibration_data = CalibrationDataLoader(resolution)
                    int8_cal_path = self.openvino_converter.to_int8_calibrated(
                        onnx_path, model_name, resolution, calibration_data
                    )
                    results.append(ExportResult(
                        model_name, resolution, ExportFormat.INT8_CALIBRATED,
                        int8_cal_path, True
                    ))
                except FileNotFoundError as e:
                    print(f"   WARNING: {e}")
                    results.append(ExportResult(
                        model_name, resolution, ExportFormat.INT8_CALIBRATED,
                        Path(), False, str(e)
                    ))

            print(f"\n{model_name} @ {resolution}px exported successfully")

        except Exception as e:
            print(f"\nError exporting {model_name} @ {resolution}px: {e}")
            results.append(ExportResult(
                model_name, resolution, formats[0] if formats else ExportFormat.ONNX,
                Path(), False, str(e)
            ))
            import traceback
            traceback.print_exc()

        return results

    def export_batch(
        self,
        yolo_version: str,
        sizes: List[str],
        task: str,
        resolutions: List[int],
        formats: List[ExportFormat],
        dry_run: bool = False,
    ) -> List[ExportResult]:
        """
        Export multiple model configurations.

        Args:
            yolo_version: YOLO version
            sizes: List of model sizes
            task: Task type
            resolutions: List of resolutions
            formats: List of formats
            dry_run: If True, only show what would be exported

        Returns:
            List of all ExportResult objects
        """
        all_results = []

        total = len(sizes) * len(resolutions)
        print(f"\nExport Plan:")
        print(f"   YOLO: {yolo_version}")
        print(f"   Task: {task}")
        print(f"   Sizes: {', '.join(sizes)}")
        print(f"   Resolutions: {', '.join(map(str, resolutions))}")
        print(f"   Formats: {', '.join(f.value for f in formats)}")
        print(f"   Total: {total} models")

        for size in sizes:
            for resolution in resolutions:
                results = self.export(
                    yolo_version, size, task, resolution,
                    formats, dry_run
                )
                all_results.extend(results)

        # Summary
        successful = sum(1 for r in all_results if r.success)
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Exported: {successful}/{len(all_results)} models")

        return all_results
