"""
Bakery - INT8 Export Pipeline V2
=================================
Exporta modelos YOLO a OpenVINO IR optimizados para CPU con VNNI.

VERSIÓN 2: Usa OpenVINO Python API directamente (sin Model Optimizer CLI)
- Convierte ONNX a OpenVINO IR usando ov.Core() y ov.save_model()
- Más simple y confiable que usar Model Optimizer CLI
- Para INT8 real con calibración, usar calibrate_int8.py después

Filosofía: "Complejidad por diseño, no por accidente"
- INT8 para CPU (VNNI acceleration)
- Export flexible por tamaño y resolución
- Batch processing de múltiples configuraciones
- Compatible con pipeline dual (segmentation + pose)

Estrategia:
-----------
INT8 en CPU (VNNI): Ideal para edge devices, baja latencia, bajo consumo
- Resoluciones bajas (320px): Máximo speedup con VNNI (~2-4x vs FP32)
- Resoluciones medias (640px): Buen balance velocidad/precisión
- Optimizado para balanceo de carga CPU/GPU en pipelines duales

Requisitos:
-----------
- CPU Intel con VNNI (Ice Lake+, 2019+): Verificar con `uv run scripts/int8_vnni/verify_vnni.py`
- OpenVINO 2024.0.0+ (solo openvino, no requiere openvino-dev)
- Modelos YOLO (auto-descargados por Ultralytics si no existen)

Uso:
----
    # Exportar modelo específico
    uv run export_int8_v2.py --model yolo11n --resolution 320

    # Exportar múltiples modelos y resoluciones
    uv run export_int8_v2.py --model yolo11n yolo11s --resolution 320 640

    # Exportar por tamaño (detection)
    uv run export_int8_v2.py --model-size n s --type detection --resolution 320

    # Exportar todos los modelos detection
    uv run export_int8_v2.py --all-detection

    # Exportar todos los tipos (detection, segmentation, pose)
    uv run export_int8_v2.py --all

    # Exportar solo segmentation
    uv run export_int8_v2.py --type segmentation --model-size m --resolution 320

    # Dry-run (ver qué se exportaría sin ejecutar)
    uv run export_int8_v2.py --model yolo11n --resolution 320 --dry-run

Notas:
------
- Esta versión convierte ONNX a OpenVINO IR (FP32)
- Para cuantización INT8 real con calibración, usar: `uv run calibrate_int8.py`
- VNNI se usa automáticamente si el CPU lo tiene (no requiere configuración)
- Verificar uso real de INT8: `uv run verify_int8_execution.py --device CPU`

Referencias:
------------
- VNNI: Vector Neural Network Instructions (Intel CPU)
- INT8 Quantization: https://docs.openvino.ai/latest/ptq_introduction.html
- Documentación completa: docs/INT8_VNNI_LOAD_BALANCING.md
"""

from pathlib import Path
from typing import List, Dict
from ultralytics import YOLO
import openvino as ov
import argparse


# ============================================================================
# CONFIGURACIÓN: Modelos y Resoluciones Disponibles
# ============================================================================

# Tamaños de modelo disponibles (detection)
MODEL_SIZES_DETECTION = {
    "n": "yolo11n",
    "s": "yolo11s",
    "m": "yolo11m",
    "l": "yolo11l",
    "x": "yolo11x",
}

# Tamaños de modelo disponibles (segmentation)
MODEL_SIZES_SEGMENTATION = {
    "n": "yolo11n-seg",
    "s": "yolo11s-seg",
    "m": "yolo11m-seg",
    "l": "yolo11l-seg",
    "x": "yolo11x-seg",
}

# Tamaños de modelo disponibles (pose)
MODEL_SIZES_POSE = {
    "n": "yolo11n-pose",
    "s": "yolo11s-pose",
    "m": "yolo11m-pose",
    "l": "yolo11l-pose",
    "x": "yolo11x-pose",
}

# Resoluciones disponibles (múltiplos de 32 para YOLO)
RESOLUTIONS = [320, 640]  # Resoluciones recomendadas para INT8

# Tipos de modelo disponibles
MODEL_TYPES = {
    "detection": MODEL_SIZES_DETECTION,
    "segmentation": MODEL_SIZES_SEGMENTATION,
    "pose": MODEL_SIZES_POSE,
}


# ============================================================================
# FUNCIONES DE EXPORTACIÓN
# ============================================================================

def export_to_onnx(
    model_name: str,
    imgsz: int,
    output_dir: Path
) -> Path:
    """
    Exporta modelo YOLO a ONNX con resolución específica.

    Args:
        model_name: Nombre del modelo (ej: yolo11n, yolo11s-seg)
        imgsz: Tamaño de imagen (320, 640)
        output_dir: Directorio de salida

    Returns:
        Path al archivo ONNX generado
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"   📦 Exportando {model_name} a ONNX ({imgsz}x{imgsz})...")

    # Load & Export (YOLO auto-descarga si no existe)
    model = YOLO(model_name)
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,  # Simplifica el grafo ONNX
        dynamic=False,  # Resolución fija para mejor optimización
    )

    # Renombrar con resolución
    onnx_file = Path(export_path)
    new_name = f"{model_name}_{imgsz}.onnx"
    target_path = output_dir / new_name

    # Mover si no está en el directorio correcto
    if onnx_file != target_path:
        onnx_file.rename(target_path)

    print(f"   ✅ ONNX exportado: {target_path.name}")
    return target_path


def convert_to_openvino_ir(onnx_path: Path, output_dir: Path) -> Path:
    """
    Convierte modelo ONNX a OpenVINO IR usando OpenVINO Python API.

    Args:
        onnx_path: Ruta al archivo ONNX (FP32)
        output_dir: Directorio para modelo OpenVINO IR

    Returns:
        Path al directorio del modelo OpenVINO IR

    Note:
        Esta función convierte ONNX a OpenVINO IR (FP32) usando la API de Python.
        Para cuantización INT8 real con calibración, usar calibrate_int8.py después.
        OpenVINO puede ejecutar algunas operaciones en INT8 automáticamente si el
        CPU tiene VNNI, pero el modelo IR sigue siendo FP32.
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX no encontrado: {onnx_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"   🔧 Convirtiendo {onnx_path.name} a OpenVINO IR...")

    # Nombre del modelo
    model_name = onnx_path.stem + "_int8"
    output_path = output_dir / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Paso 1: Cargar modelo ONNX con OpenVINO
        print(f"   ⚙️  Cargando modelo ONNX...")
        core = ov.Core()
        model = core.read_model(str(onnx_path))

        # Paso 2: Guardar como OpenVINO IR
        xml_file = output_path / f"{model_name}.xml"
        print(f"   ⚙️  Guardando modelo OpenVINO IR...")
        ov.save_model(model, str(xml_file))

    except Exception as e:
        raise RuntimeError(
            f"Error convirtiendo modelo a OpenVINO IR:\n{e}\n\n"
            f"💡 Asegúrate de tener openvino instalado: uv pip install openvino"
        )

    # Verificar que se generó el archivo
    if not xml_file.exists():
        raise RuntimeError(f"Modelo OpenVINO IR no generado: {xml_file}")

    print(f"   ✅ OpenVINO IR exportado: {output_path.name}/")
    print(f"   💡 Para cuantización INT8 con calibración, usar: uv run calibrate_int8.py")
    return output_path


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def export_model_config(
    model_name: str,
    model_type: str,
    resolution: int,
    exports_base: Path,
    dry_run: bool = False
) -> List[Path]:
    """
    Exporta modelo con tipo y resolución específicos.

    Args:
        model_name: Nombre del modelo (ej: yolo11n, yolo11s-seg)
        model_type: Tipo de modelo (detection, segmentation, pose)
        resolution: Resolución del modelo (320, 640)
        exports_base: Directorio base para exports
        dry_run: Si True, solo muestra qué se exportaría

    Returns:
        Lista de paths exportados
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Resolución inválida: {resolution}. Opciones: {RESOLUTIONS}")

    print(f"\n{'='*70}")
    print(f"🎯 Exportando: {model_name} @ {resolution}x{resolution} (OpenVINO IR)")
    print(f"{'='*70}")
    print(f"   Modelo: {model_name}")
    print(f"   Tipo: {model_type}")
    print(f"   Resolución: {resolution}x{resolution}")
    print(f"   Precisión: OpenVINO IR (FP32, optimizado para CPU/VNNI)")

    if dry_run:
        print(f"   🏃 DRY-RUN: No se exportará nada")
        return []

    exported_paths = []

    try:
        # Directorios
        FP32_DIR = exports_base / "fp32" / model_type / model_name
        INT8_DIR = exports_base / "int8" / model_type / model_name

        # Paso 1: Export ONNX (FP32)
        print(f"\n📦 Paso 1/2: Exportando ONNX (FP32)...")
        onnx_path = export_to_onnx(model_name, resolution, FP32_DIR)
        exported_paths.append(onnx_path)

        # Paso 2: Convert to OpenVINO IR
        print(f"\n🔧 Paso 2/2: Convirtiendo a OpenVINO IR...")
        ir_path = convert_to_openvino_ir(onnx_path, INT8_DIR)
        exported_paths.append(ir_path)

        print(f"\n✅ {model_name} @ {resolution}px (OpenVINO IR) exportado correctamente")

    except Exception as e:
        print(f"\n❌ Error exportando {model_name} @ {resolution}px: {e}")
        import traceback
        traceback.print_exc()

    return exported_paths


def main():
    """Pipeline principal: Export flexible para modelos OpenVINO IR"""

    parser = argparse.ArgumentParser(
        description="Export pipeline flexible para modelos YOLO OpenVINO IR (CPU/VNNI) - V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

  # Exportar modelo específico
  uv run export_int8_v2.py --model yolo11n --resolution 320

  # Exportar múltiples modelos y resoluciones
  uv run export_int8_v2.py --model yolo11n yolo11s --resolution 320 640

  # Exportar por tamaño (detection)
  uv run export_int8_v2.py --model-size n s --resolution 320

  # Exportar todos los modelos detection
  uv run export_int8_v2.py --all-detection

  # Exportar todos los tipos (detection, segmentation, pose)
  uv run export_int8_v2.py --all

  # Exportar solo segmentation
  uv run export_int8_v2.py --type segmentation --model-size m --resolution 320

  # Dry-run (ver qué se exportaría)
  uv run export_int8_v2.py --model yolo11n --resolution 320 --dry-run

Tamaños disponibles:
  n: YOLO11n (muy rápido, baja precisión)
  s: YOLO11s (rápido, balance razonable)
  m: YOLO11m (balanceado, buena calidad)
  l: YOLO11l (lento, alta precisión)
  x: YOLO11x (muy lento, máxima precisión)

Resoluciones disponibles:
  320: Resolución estándar baja (recomendado para INT8/VNNI)
  640: Resolución media (balanceado)

Nota: Esta versión crea modelos OpenVINO IR (FP32). Para INT8 real, usar calibrate_int8.py después.
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        help="Nombre(s) del modelo (ej: yolo11n, yolo11s-seg). Puede especificar múltiples."
    )
    parser.add_argument(
        "--model-size",
        type=str,
        nargs="+",
        choices=["n", "s", "m", "l", "x"],
        default=None,
        help="Tamaño(s) del modelo (n, s, m, l, x). Requiere --type."
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["detection", "segmentation", "pose"],
        default="detection",
        help="Tipo de modelo (default: detection)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        choices=RESOLUTIONS,
        default=None,
        help="Resolución(es) del modelo (320, 640). Puede especificar múltiples."
    )
    parser.add_argument(
        "--all-detection",
        action="store_true",
        help="Exportar todos los modelos detection (n, s, m, l, x) en todas las resoluciones"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Exportar todos los tipos de modelos en todas las resoluciones"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostrar qué se exportaría sin ejecutar"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exports",
        help="Directorio base para exports (default: exports/)"
    )

    args = parser.parse_args()

    # Configuración
    EXPORTS_BASE = Path(args.output)

    print("🎯 Bakery - Export Pipeline OpenVINO IR V2 (CPU/VNNI)")
    print("=" * 70)

    # Determinar qué exportar
    models_to_export = []
    resolutions_to_export = args.resolution or RESOLUTIONS

    if args.all:
        # Exportar todos los tipos
        print("📦 Modo: EXPORTAR TODOS LOS TIPOS")
        for model_type, model_sizes in MODEL_TYPES.items():
            for size in model_sizes.keys():
                for resolution in resolutions_to_export:
                    model_name = model_sizes[size]
                    models_to_export.append((model_name, model_type, resolution))

    elif args.all_detection:
        # Exportar todos los detection
        print("📦 Modo: EXPORTAR TODOS LOS DETECTION")
        for size in MODEL_SIZES_DETECTION.keys():
            for resolution in resolutions_to_export:
                model_name = MODEL_SIZES_DETECTION[size]
                models_to_export.append((model_name, "detection", resolution))

    elif args.model:
        # Modelos específicos
        print("📦 Modo: EXPORTACIÓN POR NOMBRE DE MODELO")
        for model_name in args.model:
            # Detectar tipo del nombre
            if "-seg" in model_name:
                model_type = "segmentation"
            elif "-pose" in model_name:
                model_type = "pose"
            else:
                model_type = "detection"

            for resolution in resolutions_to_export:
                models_to_export.append((model_name, model_type, resolution))

    elif args.model_size:
        # Modelos por tamaño
        if not args.type:
            print("❌ Error: --model-size requiere --type")
            return

        print(f"📦 Modo: EXPORTACIÓN POR TAMAÑO ({args.type})")
        model_sizes = MODEL_TYPES.get(args.type, {})
        for size in args.model_size:
            if size not in model_sizes:
                print(f"⚠️  Tamaño {size} no disponible para tipo {args.type}")
                continue
            model_name = model_sizes[size]
            for resolution in resolutions_to_export:
                models_to_export.append((model_name, args.type, resolution))

    else:
        print("❌ Error: Debes especificar --model, --model-size, --all-detection, o --all")
        print("\nEjemplos:")
        print("  uv run export_int8_v2.py --model yolo11n --resolution 320")
        print("  uv run export_int8_v2.py --model-size n s --type detection --resolution 320")
        print("  uv run export_int8_v2.py --all-detection")
        return

    # Mostrar plan
    print("=" * 70)
    print("\n📋 PLAN DE EXPORTACIÓN:")
    print(f"   Total combinaciones: {len(models_to_export)}")
    if len(models_to_export) <= 10:
        for model_name, model_type, resolution in models_to_export:
            print(f"   - {model_name} ({model_type}) @ {resolution}px")

    if args.dry_run:
        print("\n🏃 Modo DRY-RUN: No se exportará nada")

    print("\n" + "=" * 70)

    # Export por combinación
    all_exports = {}
    for model_name, model_type, resolution in models_to_export:
        combo_name = f"{model_name}@{resolution}"
        exported = export_model_config(
            model_name, model_type, resolution,
            EXPORTS_BASE, args.dry_run
        )
        all_exports[combo_name] = exported

    # Resumen final
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE EXPORTACIÓN")
    print("=" * 70)

    total_files = sum(len(paths) for paths in all_exports.values())
    print(f"\n✅ Total de archivos exportados: {total_files}")

    if not args.dry_run:
        print("\n📂 Estructura generada:")
        print(f"   {EXPORTS_BASE}/")
        print("   ├── fp32/")
        print("   │   ├── detection/")
        print("   │   ├── segmentation/")
        print("   │   └── pose/")
        print("   └── int8/")
        print("       ├── detection/")
        print("       ├── segmentation/")
        print("       └── pose/")

        print("\n💡 Próximos pasos:")
        print("   # Verificar VNNI en CPU")
        print("   uv run scripts/int8_vnni/verify_vnni.py")
        print("\n   # Verificar ejecución OpenVINO IR")
        print("   uv run verify_int8_execution.py --device CPU")
        print("\n   # Para cuantización INT8 con calibración:")
        print("   uv run calibrate_int8.py --model yolo11n --resolution 320")

    print("\n" + "=" * 70)
    print("🎉 Exportación completada!")
    print("=" * 70)


if __name__ == "__main__":
    main()
