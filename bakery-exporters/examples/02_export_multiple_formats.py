"""
Example 2: Export Multiple Formats

Export a model to ONNX, FP16, and INT8 formats using the pipeline.

Usage:
    python examples/02_export_multiple_formats.py
"""

from bakery_exporters import ExportPipeline, ExportFormat
from bakery.catalog import config


def main():
    print("Bakery Exporters - Example 2: Export Multiple Formats")
    print("=" * 60)

    # Create pipeline
    pipeline = ExportPipeline()

    # Export configuration
    yolo_version = "26"
    size = "n"
    task = "segmentation"
    resolution = 320
    formats = [
        ExportFormat.ONNX,
        ExportFormat.FP16,
        ExportFormat.INT8
    ]

    print(f"\nExporting: yolo{yolo_version}{size}-{task}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Formats: {', '.join(f.value for f in formats)}")
    print(f"Output directory: {config.models_dir}")

    # Export
    results = pipeline.export(
        yolo_version=yolo_version,
        size=size,
        task=task,
        resolution=resolution,
        formats=formats,
        dry_run=False
    )

    # Show results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    successful = 0
    for result in results:
        if result.success:
            print(f"✓ {result.format.value.upper()}")
            print(f"  Path: {result.output_path}")
            successful += 1
        else:
            print(f"✗ {result.format.value.upper()}")
            print(f"  Error: {result.error}")

    print(f"\nSummary: {successful}/{len(results)} formats exported successfully")

    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
