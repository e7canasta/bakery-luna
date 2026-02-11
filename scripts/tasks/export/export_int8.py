"""
Bakery - INT8 Export Pipeline
==============================

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

from typing import List
from ultralytics import YOLO
import openvino as ov
import nncf
import numpy as np
import argparse

from bakery.config import (
    config,
    ModelPath,
    YOLO_VERSIONS,
    MODEL_SIZES,
    RESOLUTIONS,
    TASK_TYPES,
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
    output_path = ModelPath.build(model_name, resolution, "onnx")

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


def quantize_to_int8(onnx_path: Path, model_name: str, resolution: int) -> Path:
    """
    Quantize ONNX model to INT8 using NNCF with synthetic data.

    Args:
        onnx_path: Path to ONNX file
        model_name: Full model name
        resolution: Input resolution

    Returns:
        Path to exported OpenVINO INT8 model (.xml)
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    output_path = ModelPath.build(model_name, resolution, "int8")

    print(f"   🔧 Cuantizando a INT8 (CPU/VNNI)...")

    # Load ONNX with OpenVINO
    print(f"   ⚙️  Cargando modelo ONNX...")
    core = ov.Core()
    model = core.read_model(str(onnx_path))

    # Create synthetic calibration data
    print(f"   ⚙️  Generando datos sintéticos para calibración...")
    synthetic_samples = [
        np.random.rand(1, 3, resolution, resolution).astype(np.float32)
        for _ in range(100)
    ]

    # Quantize with NNCF
    print(f"   ⚙️  Aplicando cuantización INT8 con NNCF...")
    quantized_model = nncf.quantize(
        model,
        nncf.Dataset(synthetic_samples),
        preset=nncf.QuantizationPreset.PERFORMANCE,
        subset_size=100,
    )

    # Save quantized model
    print(f"   ⚙️  Guardando modelo INT8...")
    ov.save_model(quantized_model, str(output_path))

    print(f"   ✅ INT8: {output_path.relative_to(config.models_dir)}")
    print(f"   💡 Para mejor accuracy: uv run calibrate_int8.py")
    return output_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def export_model(
    yolo_version: str,
    size: str,
    task: str,
    resolution: int,
    formats: List[str],
    dry_run: bool = False
) -> List[Path]:
    """
    Export a single model configuration.

    Args:
        yolo_version: YOLO version (11, 26, 8)
        size: Model size (n, s, m, l, x)
        task: Task type (detection, segmentation, pose)
        resolution: Input resolution
        formats: List of formats to export (onnx, int8)
        dry_run: If True, only show what would be exported

    Returns:
        List of exported paths
    """
    model_name = ModelPath.get_model_name(yolo_version, size, task)

    print(f"\n{'='*70}")
    print(f"🎯 Exportando: {model_name} @ {resolution}x{resolution} (INT8)")
    print(f"{'='*70}")
    print(f"   Modelo: {model_name}")
    print(f"   Tarea: {task}")
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

        # Step 2: Quantize to INT8 if requested
        if "int8" in formats:
            print(f"\n🔧 Paso 2: Cuantizando a INT8...")
            int8_path = quantize_to_int8(onnx_path, model_name, resolution)
            exported.append(int8_path)

        print(f"\n✅ {model_name} @ {resolution}px exportado correctamente")

    except Exception as e:
        print(f"\n❌ Error exportando {model_name} @ {resolution}px: {e}")
        import traceback
        traceback.print_exc()

    return exported


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export YOLO models to INT8 (CPU/VNNI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Ejemplos:

  # Modelo específico
  uv run export_int8.py --model 26 --model-size n --type segmentation --resolution 320

  # Múltiples tamaños
  uv run export_int8.py --model 26 --model-size n s m --type pose --resolution 320

  # Detection (default)
  uv run export_int8.py --model 11 --model-size n --resolution 320

Versiones YOLO: {', '.join(YOLO_VERSIONS)} (default: {config.default_yolo_version})
Tamaños: {', '.join(MODEL_SIZES)}
Tareas: {', '.join(TASK_TYPES)}
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
        required=True,
        help="Tamaño(s) del modelo"
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        choices=TASK_TYPES,
        default="detection",
        help="Tipo de tarea (default: detection)"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        nargs="+",
        choices=RESOLUTIONS,
        required=True,
        help="Resolución(es)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        nargs="+",
        choices=["onnx", "int8"],
        default=["int8"],
        help="Formato(s) de salida (default: int8)"
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
    task = args.type
    formats = args.format

    print("🎯 Bakery - Export Pipeline INT8 (CPU/VNNI)")
    print("=" * 70)
    print(f"   Directorio: {config.models_dir.absolute()}")

    model_sizes = args.model_size
    resolutions = args.resolution

    # Show plan
    print("=" * 70)
    print(f"\n📋 PLAN DE EXPORTACIÓN:")
    print(f"   YOLO: {yolo_version}")
    print(f"   Tarea: {task}")
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
            model_name = ModelPath.get_model_name(yolo_version, size, task)
            key = f"{model_name}@{resolution}"
            exported = export_model(yolo_version, size, task, resolution, formats, args.dry_run)
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

        print("\n💡 Próximos pasos:")
        print("   # Para mejor accuracy, calibrar con datos reales:")
        print("   uv run calibrate_int8.py")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
