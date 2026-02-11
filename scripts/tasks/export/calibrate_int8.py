"""
Bakery - INT8 Calibration with NNCF
====================================

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

import numpy as np
import argparse
from typing import List, Iterator
import openvino as ov
import nncf

from bakery.config import (
    config,
    ModelPath,
    RESOLUTIONS,
)


# ============================================================================
# CALIBRATION DATA LOADER
# ============================================================================

class CalibrationDataLoader:
    """
    DataLoader para calibración NNCF.
    Carga frames pre-procesados (.npy) para calibración.
    """

    def __init__(self, resolution: int):
        """
        Args:
            resolution: Resolución del modelo (320 o 640)
        """
        self.calibration_dir = config.calibration_dir / f"preprocessed_{resolution}"

        if not self.calibration_dir.exists():
            raise FileNotFoundError(
                f"Calibration data no encontrado: {self.calibration_dir}\n"
                f"Ejecuta primero: uv run extract_calibration_frames.py"
            )

        # Cargar todos los .npy files
        self.frame_files = sorted(self.calibration_dir.glob("*.npy"))

        if not self.frame_files:
            raise ValueError(f"No se encontraron frames en {self.calibration_dir}")

        print(f"   📦 Cargados {len(self.frame_files)} frames para calibración")

    def __len__(self):
        return len(self.frame_files)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Itera sobre frames cargándolos on-demand"""
        for frame_file in self.frame_files:
            frame = np.load(frame_file)
            yield frame


# ============================================================================
# DISCOVERY FUNCTIONS
# ============================================================================

def discover_onnx_models(
    model_name: str = None,
    resolution: int = None
) -> List[dict]:
    """
    Descubre modelos ONNX disponibles en el directorio de modelos.

    Args:
        model_name: Nombre del modelo (ej: yolo26n-seg) o None para todos
        resolution: Resolución (320, 640) o None para todas

    Returns:
        Lista de dicts con info del modelo {name, resolution, onnx_path}
    """
    models = []

    if not config.models_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {config.models_dir}")

    # Si se especifica modelo, buscar solo ese
    if model_name:
        model_dirs = [config.models_dir / model_name]
    else:
        model_dirs = [d for d in config.models_dir.iterdir() if d.is_dir()]

    for model_dir in sorted(model_dirs):
        if not model_dir.exists():
            continue

        name = model_dir.name

        # Buscar resoluciones
        for res_dir in sorted(model_dir.iterdir()):
            if not res_dir.is_dir():
                continue

            try:
                res = int(res_dir.name)
            except ValueError:
                continue

            # Filtrar por resolución si se especificó
            if resolution and res != resolution:
                continue

            # Buscar ONNX
            onnx_path = res_dir / "onnx" / "model.onnx"
            if onnx_path.exists():
                models.append({
                    "name": name,
                    "resolution": res,
                    "onnx_path": onnx_path,
                })

    return models


# ============================================================================
# CALIBRATION FUNCTION
# ============================================================================

def calibrate_model(
    model_name: str,
    resolution: int,
    onnx_path: Path
) -> Path:
    """
    Calibra modelo ONNX a INT8 usando NNCF con datos reales.

    Args:
        model_name: Nombre del modelo
        resolution: Resolución
        onnx_path: Path al modelo ONNX

    Returns:
        Path al modelo INT8 calibrado (.xml)
    """
    print(f"\n🔧 Calibrando: {model_name} @ {resolution}px")
    print("-" * 60)
    print(f"   ONNX: {onnx_path.relative_to(config.models_dir)}")

    # Get output path
    output_path = ModelPath.build(model_name, resolution, "int8_calibrated")

    # 1. Leer modelo ONNX
    print(f"   1️⃣  Leyendo modelo ONNX...")
    core = ov.Core()
    model = core.read_model(str(onnx_path))

    # 2. Preparar calibration dataloader
    print(f"   2️⃣  Preparando calibration dataset...")
    calibration_loader = CalibrationDataLoader(resolution)

    # 3. Crear calibration dataset para NNCF
    calibration_dataset = nncf.Dataset(calibration_loader)

    # 4. Cuantizar con NNCF (Post-Training Quantization)
    print(f"   3️⃣  Cuantizando a INT8 con NNCF...")
    print(f"      (Esto puede tomar 1-2 minutos...)")

    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        preset=nncf.QuantizationPreset.MIXED,  # MIXED = balance accuracy/speed
    )

    # 5. Guardar modelo cuantizado
    print(f"   4️⃣  Guardando modelo INT8 calibrado...")
    ov.save_model(quantized_model, str(output_path))

    print(f"   ✅ Calibrado: {output_path.relative_to(config.models_dir)}")

    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calibra modelos ONNX a INT8 usando NNCF con datos reales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Ejemplos:

  # Calibrar modelo específico
  uv run calibrate_int8.py --model yolo26n-seg --resolution 320

  # Calibrar todos los modelos ONNX encontrados
  uv run calibrate_int8.py --all

  # Calibrar todos con resolución específica
  uv run calibrate_int8.py --all --resolution 320

Resoluciones: {', '.join(map(str, RESOLUTIONS))}

Configuración:
  BAKERY_MODELS_DIR={config.models_dir}
  BAKERY_CALIBRATION_DIR={config.calibration_dir}

Requisitos:
  - Datos de calibración en: {config.calibration_dir}/preprocessed_{{resolution}}/
  - Ejecutar primero: uv run extract_calibration_frames.py
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nombre del modelo (ej: yolo26n-seg). Si no se especifica, usa --all"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        choices=RESOLUTIONS,
        default=None,
        help="Resolución del modelo. Si no se especifica, calibra todas."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Calibrar todos los modelos ONNX encontrados"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help=f"Override directorio de modelos (default: {config.models_dir})"
    )

    args = parser.parse_args()

    # Validate args
    if not args.model and not args.all:
        print("❌ Error: Especifica --model o --all")
        parser.print_help()
        return 1

    # Override models dir if specified
    if args.output:
        config.models_dir = Path(args.output)

    print("🎯 Bakery - INT8 Calibration with NNCF")
    print("=" * 70)
    print(f"📂 Modelos: {config.models_dir}")
    print(f"📊 Calibration data: {config.calibration_dir}")

    if args.model:
        print(f"🎯 Modelo: {args.model}")
    if args.resolution:
        print(f"📐 Resolución: {args.resolution}")

    # Validate calibration data exists
    if not config.calibration_dir.exists():
        print(f"\n❌ Error: Calibration data no encontrado en {config.calibration_dir}")
        print("\n💡 Ejecuta primero:")
        print("   uv run extract_calibration_frames.py")
        return 1

    # Discover models
    try:
        models = discover_onnx_models(args.model, args.resolution)
        print(f"\n📦 Modelos ONNX encontrados: {len(models)}")
        for m in models:
            print(f"   - {m['name']} @ {m['resolution']}px")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        return 1

    if not models:
        print("\n❌ No se encontraron modelos ONNX para calibrar")
        return 1

    print("\n" + "=" * 70)
    print(f"🔄 Calibrando {len(models)} modelos...")
    print("=" * 70)

    # Calibrate each model
    calibrated = []
    failed = []

    for idx, model_info in enumerate(models, 1):
        try:
            print(f"\n[{idx}/{len(models)}]")
            output_path = calibrate_model(
                model_info['name'],
                model_info['resolution'],
                model_info['onnx_path']
            )
            calibrated.append(output_path)

        except Exception as e:
            print(f"   ❌ Error calibrando {model_info['name']}: {e}")
            failed.append(model_info['name'])
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)
    print(f"✅ Calibrados: {len(calibrated)}")
    print(f"❌ Errores: {len(failed)}")

    if calibrated:
        print(f"\n📂 Modelos calibrados en: {config.models_dir}/")
        for path in calibrated[:5]:
            rel = path.relative_to(config.models_dir)
            print(f"   - {rel}")
        if len(calibrated) > 5:
            print(f"   ... y {len(calibrated) - 5} más")

    if failed:
        print(f"\n⚠️  Modelos que fallaron:")
        for name in failed:
            print(f"   - {name}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
