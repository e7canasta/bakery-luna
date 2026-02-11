"""
Bakery - INT8 Calibration with NNCF
====================================

Thin wrapper around bakery-exporters for INT8 calibration.

Calibra modelos a INT8 usando NNCF con datos de calibración reales.

Estructura de entrada/salida (configurable via BAKERY_MODELS_DIR o .env):
    models/
    └── yolo26n-seg/
        └── 320/
            ├── onnx/model.onnx          <- entrada
            └── int8_calibrated/model.xml <- salida

Uso:
    # Calibrar modelo específico
    uv run calibrate_int8.py --model yolo26n-seg --resolution 320

    # Calibrar todos los modelos ONNX encontrados
    uv run calibrate_int8.py --all

    # Calibrar con resolución específica
    uv run calibrate_int8.py --all --resolution 320

Configuración:
    Variables de entorno (o .env):
    - BAKERY_MODELS_DIR: Directorio de modelos (default: models)
    - BAKERY_CALIBRATION_DIR: Directorio de calibración (default: calibration_data)

Requisitos:
    - Datos de calibración pre-procesados en BAKERY_CALIBRATION_DIR/preprocessed_{resolution}/
    - Ejecutar primero: uv run extract_calibration_frames.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import openvino as ov

from bakery_catalog import config, RESOLUTIONS
from bakery_exporters import ExportPipeline, ExportFormat
from bakery_exporters.calibration import discover_onnx_models, CalibrationDataLoader


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calibrate YOLO models to INT8 using NNCF with real data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:

  # Calibrate specific model
  uv run calibrate_int8.py --model yolo26n-seg --resolution 320

  # Calibrate all ONNX models found
  uv run calibrate_int8.py --all

  # Calibrate all with specific resolution
  uv run calibrate_int8.py --all --resolution 320

Resolutions: {', '.join(map(str, RESOLUTIONS))}

Configuration:
  BAKERY_MODELS_DIR={config.models_dir}
  BAKERY_CALIBRATION_DIR={config.calibration_dir}

Requirements:
  - Calibration data in: {config.calibration_dir}/preprocessed_{{resolution}}/
  - Run first: uv run extract_calibration_frames.py
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., yolo26n-seg). If not specified, use --all"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        choices=RESOLUTIONS,
        default=None,
        help="Model resolution. If not specified, calibrate all."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Calibrate all ONNX models found"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help=f"Override models directory (default: {config.models_dir})"
    )

    args = parser.parse_args()

    # Validate args
    if not args.model and not args.all:
        print("Error: Specify --model or --all")
        parser.print_help()
        return 1

    # Override config if specified
    if args.output:
        config.models_dir = Path(args.output)

    print("🎯 Bakery - INT8 Calibration with NNCF")
    print("=" * 70)
    print(f"📂 Models: {config.models_dir}")
    print(f"📊 Calibration data: {config.calibration_dir}")

    if args.model:
        print(f"🎯 Model: {args.model}")
    if args.resolution:
        print(f"📐 Resolution: {args.resolution}")

    # Validate calibration data exists
    if not config.calibration_dir.exists():
        print(f"\nError: Calibration data not found in {config.calibration_dir}")
        print("\n💡 Run first:")
        print("   uv run extract_calibration_frames.py")
        return 1

    # Discover models
    try:
        models = discover_onnx_models(args.model, args.resolution)
        print(f"\n📦 ONNX models found: {len(models)}")
        for m in models:
            print(f"   - {m['name']} @ {m['resolution']}px")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        return 1

    if not models:
        print("\nError: No ONNX models found to calibrate")
        return 1

    print("\n" + "=" * 70)
    print(f"🔄 Calibrating {len(models)} models...")
    print("=" * 70)

    # Calibrate each model
    calibrated = []
    failed = []

    for idx, model_info in enumerate(models, 1):
        try:
            print(f"\n[{idx}/{len(models)}]")
            print(f"\nCalibrating: {model_info['name']} @ {model_info['resolution']}px")
            print("-" * 60)
            print(f"   ONNX: {model_info['onnx_path'].relative_to(config.models_dir)}")

            # Load model ONNX
            print(f"   1️⃣  Reading ONNX model...")
            core = ov.Core()
            model = core.read_model(str(model_info['onnx_path']))

            # Prepare calibration dataset
            print(f"   2️⃣  Preparing calibration dataset...")
            calibration_loader = CalibrationDataLoader(model_info['resolution'])
            calibration_dataset = __import__('nncf').Dataset(calibration_loader)

            # Quantize with NNCF
            print(f"   3️⃣  Quantizing to INT8 with NNCF...")
            print(f"      (This may take 1-2 minutes...)")

            nncf = __import__('nncf')
            quantized_model = nncf.quantize(
                model,
                calibration_dataset,
                preset=nncf.QuantizationPreset.MIXED,
            )

            # Save quantized model
            print(f"   4️⃣  Saving INT8 calibrated model...")
            from bakery_catalog import ModelPath
            output_path = ModelPath.build(
                model_info['name'],
                model_info['resolution'],
                "int8_calibrated"
            )
            ov.save_model(quantized_model, str(output_path))

            print(f"   ✅ Calibrated: {output_path.relative_to(config.models_dir)}")
            calibrated.append(output_path)

        except Exception as e:
            print(f"   Error calibrating {model_info['name']}: {e}")
            failed.append(model_info['name'])
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"✅ Calibrated: {len(calibrated)}")
    print(f"❌ Failed: {len(failed)}")

    if calibrated:
        print(f"\n📂 Calibrated models in: {config.models_dir}/")
        for path in calibrated[:5]:
            rel = path.relative_to(config.models_dir)
            print(f"   - {rel}")
        if len(calibrated) > 5:
            print(f"   ... and {len(calibrated) - 5} more")

    if failed:
        print(f"\n⚠️  Models that failed:")
        for name in failed:
            print(f"   - {name}")

    print("\n" + "=" * 70)
    return 0 if calibrated else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
