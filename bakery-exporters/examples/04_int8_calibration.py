"""
Example 4: INT8 Calibration with Real Data

Export and calibrate a model to INT8 using real calibration data.

Usage:
    python examples/04_int8_calibration.py

Prerequisites:
    - Calibration data must be prepared first:
      python extract_calibration_frames.py --resolution 320
"""

import sys
from pathlib import Path

from bakery_exporters import (
    ExportPipeline, ExportFormat,
    OpenVINOConverter,
    CalibrationDataLoader
)
from bakery.catalog import config, ModelPath


def main():
    print("Bakery Exporters - Example 4: INT8 Calibration")
    print("=" * 60)

    model_name = "yolo26n-seg"
    resolution = 320

    # Step 1: Check if calibration data exists
    print("\nStep 1: Checking calibration data...")
    try:
        loader = CalibrationDataLoader(resolution)
        print(f"✓ Found {len(loader)} calibration frames")
    except FileNotFoundError as e:
        print(f"✗ Calibration data not found: {e}")
        print("\nPlease prepare calibration data first:")
        print("  python extract_calibration_frames.py --resolution 320")
        return 1

    # Step 2: Export ONNX
    print(f"\nStep 2: Exporting {model_name} to ONNX...")
    pipeline = ExportPipeline()

    try:
        results = pipeline.export(
            yolo_version="26",
            size="n",
            task="segmentation",
            resolution=resolution,
            formats=[ExportFormat.ONNX]
        )

        if not results[0].success:
            print(f"✗ ONNX export failed: {results[0].error}")
            return 1

        onnx_path = results[0].output_path
        print(f"✓ ONNX exported: {onnx_path.relative_to(config.models_dir)}")

    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    # Step 3: Calibrate INT8
    print(f"\nStep 3: Calibrating to INT8 with real data...")
    print(f"(This may take 1-2 minutes...)")

    try:
        converter = OpenVINOConverter()

        int8_path = converter.to_int8_calibrated(
            onnx_path,
            model_name,
            resolution,
            calibration_data=loader,
            preset="MIXED"
        )

        print(f"✓ INT8 calibrated: {int8_path.relative_to(config.models_dir)}")

    except Exception as e:
        print(f"✗ Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    print(f"\nModel: {model_name}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"\nExported formats:")
    onnx = ModelPath.get(model_name, resolution, "onnx")
    int8_cal = ModelPath.get(model_name, resolution, "int8_calibrated")

    print(f"  ONNX:   {onnx.relative_to(config.models_dir)}")
    print(f"  INT8:   {int8_cal.relative_to(config.models_dir)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
