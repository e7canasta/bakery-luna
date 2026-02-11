"""
Bakery - Sauron Eye Export Pipeline (Segmentation)
===================================================

Exporta modelos de segmentación YOLO a formatos optimizados.

Estructura de salida (configurable via BAKERY_MODELS_DIR o .env):
    models/
    └── yolo26n-seg/
        └── 320/
            ├── onnx/model.onnx
            └── fp16/model.xml + model.bin

Uso:
    # Exportar modelo específico
    uv run export_sauron_segmentation.py --model 26 --model-size n --resolution 320

    # Exportar múltiples combinaciones
    uv run export_sauron_segmentation.py --model 26 --model-size n s m --resolution 256 320

    # Exportar todo
    uv run export_sauron_segmentation.py --model 26 --all

    # Dry-run
    uv run export_sauron_segmentation.py --model 26 --model-size n --resolution 320 --dry-run

Configuración:
    Variables de entorno (o .env):
    - BAKERY_MODELS_DIR: Directorio de modelos (default: models)
    - BAKERY_DEFAULT_YOLO: Versión YOLO default (default: 11)
    - BAKERY_ONNX_OPSET: Opset ONNX (default: 17)
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from typing import List
from ultralytics import YOLO
import openvino as ov
import argparse

from bakery.config import (
    config,
    ModelPath,
    YOLO_VERSIONS,
    MODEL_SIZES,
    RESOLUTIONS,
)


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_to_onnx(model_name: str, resolution: int) -> Path:
    """
    Export YOLO model to ONNX format.

    Args:
        model_name: Full model name (e.g., yolo26n-seg)
        resolution: Input resolution (e.g., 320)

    Returns:
        Path to exported ONNX file
    """
    # Get output path using ModelPath
    output_path = ModelPath.build(model_name, resolution, "onnx")
    output_dir = output_path.parent

    print(f"   📦 Exportando {model_name} a ONNX ({resolution}x{resolution})...")

    # Load & Export
    model = YOLO(model_name)
    export_path = model.export(
        format="onnx",
        imgsz=resolution,
        simplify=True,
        dynamic=False,
        opset=config.onnx_opset,
    )

    # Move to target location
    onnx_file = Path(export_path)
    if onnx_file != output_path:
        if output_path.exists():
            output_path.unlink()
        onnx_file.rename(output_path)

    print(f"   ✅ ONNX: {output_path.relative_to(config.models_dir)}")
    return output_path


def convert_to_fp16(onnx_path: Path, model_name: str, resolution: int) -> Path:
    """
    Convert ONNX model to FP16 OpenVINO IR format.

    Args:
        onnx_path: Path to ONNX file
        model_name: Full model name
        resolution: Input resolution

    Returns:
        Path to exported OpenVINO model (.xml)
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    # Get output path
    output_path = ModelPath.build(model_name, resolution, "fp16")

    print(f"   🔧 Convirtiendo a FP16 (GPU)...")

    # Convert
    core = ov.Core()
    model = core.read_model(onnx_path)

    # Save with FP16 compression
    ov.save_model(model, output_path, compress_to_fp16=True)

    print(f"   ✅ FP16: {output_path.relative_to(config.models_dir)}")
    return output_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def export_model(
    yolo_version: str,
    size: str,
    resolution: int,
    formats: List[str],
    dry_run: bool = False
) -> List[Path]:
    """
    Export a single model configuration.

    Args:
        yolo_version: YOLO version (11, 26, 8)
        size: Model size (n, s, m, l, x)
        resolution: Input resolution
        formats: List of formats to export (onnx, fp16)
        dry_run: If True, only show what would be exported

    Returns:
        List of exported paths
    """
    model_name = ModelPath.get_model_name(yolo_version, size, "segmentation")

    print(f"\n{'='*70}")
    print(f"🎯 Exportando: {model_name} @ {resolution}x{resolution}")
    print(f"{'='*70}")
    print(f"   Modelo: {model_name}")
    print(f"   Resolución: {resolution}x{resolution}")
    print(f"   Formatos: {', '.join(formats)}")
    print(f"   Destino: {config.models_dir}/{model_name}/{resolution}/")

    if dry_run:
        print(f"   🏃 DRY-RUN: No se exportará nada")
        return []

    exported = []

    try:
        # Step 1: Export ONNX (always needed as intermediate)
        print(f"\n📦 Paso 1: Exportando ONNX...")
        onnx_path = export_to_onnx(model_name, resolution)

        if "onnx" in formats:
            exported.append(onnx_path)

        # Step 2: Convert to FP16 if requested
        if "fp16" in formats:
            print(f"\n🔧 Paso 2: Convirtiendo a FP16...")
            fp16_path = convert_to_fp16(onnx_path, model_name, resolution)
            exported.append(fp16_path)

        print(f"\n✅ {model_name} @ {resolution}px exportado correctamente")

    except Exception as e:
        print(f"\n❌ Error exportando {model_name} @ {resolution}px: {e}")
        import traceback
        traceback.print_exc()

    return exported


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export YOLO Segmentation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Ejemplos:

  # YOLO11 (default)
  uv run export_sauron_segmentation.py --model-size m --resolution 256

  # YOLO26
  uv run export_sauron_segmentation.py --model 26 --model-size m --resolution 256

  # Múltiples combinaciones
  uv run export_sauron_segmentation.py --model 26 --model-size n s m --resolution 256 320

  # Todo
  uv run export_sauron_segmentation.py --model 26 --all

Versiones YOLO: {', '.join(YOLO_VERSIONS)} (default: {config.default_yolo_version})
Tamaños: {', '.join(MODEL_SIZES)}
Resoluciones: {', '.join(map(str, RESOLUTIONS))}

Configuración:
  BAKERY_MODELS_DIR={config.models_dir}
  BAKERY_ONNX_OPSET={config.onnx_opset}
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=YOLO_VERSIONS,
        default=config.default_yolo_version,
        help=f"Versión de YOLO (default: {config.default_yolo_version})"
    )
    parser.add_argument(
        "--model-size", "-s",
        type=str,
        nargs="+",
        choices=MODEL_SIZES,
        help="Tamaño(s) del modelo"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        nargs="+",
        choices=RESOLUTIONS,
        help="Resolución(es)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        nargs="+",
        choices=["onnx", "fp16"],
        default=["fp16"],
        help="Formato(s) de salida (default: fp16)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Exportar todas las combinaciones"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo mostrar qué se exportaría"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help=f"Override directorio de modelos (default: {config.models_dir})"
    )

    args = parser.parse_args()

    # Override models dir if specified
    if args.output:
        config.models_dir = Path(args.output)

    yolo_version = args.model
    formats = args.format

    print("🎯 Bakery - Export Pipeline (Segmentation)")
    print("=" * 70)
    print(f"   Directorio: {config.models_dir.absolute()}")

    # Determine what to export
    if args.all:
        model_sizes = MODEL_SIZES
        resolutions = RESOLUTIONS
        print("📦 Modo: TODAS LAS COMBINACIONES")
    else:
        if not args.model_size or not args.resolution:
            print("❌ Error: Especifica --model-size y --resolution, o usa --all")
            parser.print_help()
            return 1

        model_sizes = args.model_size
        resolutions = args.resolution
        print("📦 Modo: EXPORTACIÓN SELECTIVA")

    # Show plan
    print("=" * 70)
    print(f"\n📋 PLAN DE EXPORTACIÓN:")
    print(f"   YOLO: {yolo_version}")
    print(f"   Tamaños: {', '.join(model_sizes)}")
    print(f"   Resoluciones: {', '.join(map(str, resolutions))}")
    print(f"   Formatos: {', '.join(formats)}")
    print(f"   Total: {len(model_sizes) * len(resolutions)} modelos")

    if args.dry_run:
        print("\n🏃 Modo DRY-RUN")

    print("\n" + "=" * 70)

    # Export
    all_exports = {}
    for size in model_sizes:
        for resolution in resolutions:
            model_name = ModelPath.get_model_name(yolo_version, size, "segmentation")
            key = f"{model_name}@{resolution}"
            exported = export_model(yolo_version, size, resolution, formats, args.dry_run)
            all_exports[key] = exported

    # Summary
    print("\n" + "=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)

    total = sum(len(p) for p in all_exports.values())
    print(f"\n✅ Archivos exportados: {total}")

    if not args.dry_run and total > 0:
        print(f"\n📂 Estructura:")
        print(f"   {config.models_dir}/")
        for key in all_exports:
            model, res = key.split("@")
            print(f"   └── {model}/{res}/")
            for fmt in formats:
                ext = "model.onnx" if fmt == "onnx" else "model.xml"
                print(f"       └── {fmt}/{ext}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
