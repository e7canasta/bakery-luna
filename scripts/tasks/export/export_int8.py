"""
Bakery - INT8 Export Pipeline
==============================

Thin wrapper around bakery-exporters for INT8 quantization.

Exporta modelos YOLO a INT8 optimizados para CPU con VNNI.

Estructura de salida (configurable via BAKERY_MODELS_DIR o .env):
    models/
    └── yolo26n-seg/
        └── 320/
            ├── onnx/model.onnx
            └── int8/model.xml + model.bin

Uso:
    # Exportar modelo específico
    uv run export_int8.py --model 26 --model-size n --type segmentation --resolution 320

    # Exportar múltiples
    uv run export_int8.py --model 26 --model-size n s m --type pose --resolution 320

    # Dry-run
    uv run export_int8.py --model 26 --model-size n --type segmentation --resolution 320 --dry-run

Configuración:
    Variables de entorno (o .env):
    - BAKERY_MODELS_DIR: Directorio de modelos (default: models)
    - BAKERY_DEFAULT_YOLO: Versión YOLO default (default: 11)
    - BAKERY_ONNX_OPSET: Opset ONNX (default: 17)

Notas:
    - INT8 usa datos sintéticos para calibración rápida
    - Para mejor accuracy usar calibrate_int8.py con datos reales
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from bakery_catalog import config, YOLO_VERSIONS, MODEL_SIZES, RESOLUTIONS, TASK_TYPES
from bakery_exporters import ExportPipeline, ExportFormat
from bakery_exporters.cli import create_base_parser, add_format_argument


def main():
    """Main entry point."""
    parser = create_base_parser(
        "Export YOLO models to INT8 (CPU/VNNI)",
    )
    add_format_argument(parser, ["onnx", "int8"], ["int8"])

    args = parser.parse_args()

    # Override config if specified
    if args.output:
        config.models_dir = Path(args.output)

    # Validate arguments
    if not args.all and (not args.model_size or not args.resolution):
        print("Error: Specify --model-size and --resolution, or use --all")
        parser.print_help()
        return 1

    yolo_version = args.model
    task = args.type
    formats = [ExportFormat(f) for f in args.format]

    # Determine what to export
    if args.all:
        model_sizes = MODEL_SIZES
        resolutions = RESOLUTIONS
    else:
        model_sizes = args.model_size
        resolutions = args.resolution

    print("🎯 Bakery - Export Pipeline INT8 (CPU/VNNI)")
    print("=" * 70)
    print(f"   Directory: {config.models_dir.absolute()}")

    # Run pipeline
    pipeline = ExportPipeline()
    all_results = pipeline.export_batch(
        yolo_version,
        model_sizes,
        task,
        resolutions,
        formats,
        args.dry_run
    )

    print("\n" + "=" * 70)
    print("\nTIP: For better accuracy, calibrate with real data:")
    print("   uv run calibrate_int8.py --all --resolution 320")

    return 0 if any(r.success for r in all_results) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
