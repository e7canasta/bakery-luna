"""
Example 1: Basic Export

Simple example of exporting a YOLO model to ONNX format.

Usage:
    python examples/01_basic_export.py
"""

from bakery_exporters import OnnxExporter
from bakery.catalog import config, ModelPath


def main():
    print("Bakery Exporters - Example 1: Basic ONNX Export")
    print("=" * 60)

    # Create exporter
    exporter = OnnxExporter()

    # Export model
    model_name = "yolo26n-seg"
    resolution = 320

    print(f"\nExporting {model_name} to ONNX...")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Output directory: {config.models_dir}")

    try:
        onnx_path = exporter.export(model_name, resolution)
        print(f"\n✓ Success!")
        print(f"  Model exported to: {onnx_path}")
        print(f"  Relative path: {onnx_path.relative_to(config.models_dir)}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
