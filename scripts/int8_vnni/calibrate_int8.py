"""
Bakery - INT8 Calibration with NNCF
====================================
Cuantiza modelos ONNX a INT8 usando NNCF (Neural Network Compression Framework).

Filosofía: "Complejidad por diseño, no por accidente"
- Post-Training Quantization (PTQ) con calibración
- Dataset representativo pre-procesado
- Métricas de accuracy loss
- Batch processing de múltiples modelos

Referencias:
- NNCF: https://github.com/openvinotoolkit/nncf
- OpenVINO PTQ: https://docs.openvino.ai/latest/ptq_introduction.html

Uso:
    # Calibrar todos los modelos detection (yolov11n, yolov11s, yolov11m, yolov11l)
    uv run calibrate_int8.py

    # Calibrar modelo específico
    uv run calibrate_int8.py --model yolov11n

    # Calibrar resolución específica
    uv run calibrate_int8.py --resolution 320

    # Calibrar modelo + resolución específica
    uv run calibrate_int8.py --model yolov11n --resolution 640
"""

from pathlib import Path
import numpy as np
import argparse
from typing import List, Iterator
import openvino as ov
import nncf


def discover_onnx_models(
    base_dir: Path,
    model_name: str = None,
    resolution: int = None,
    exclude_patterns: list = None
) -> List[Path]:
    """
    Descubre modelos ONNX disponibles en exports/fp32/.

    Args:
        base_dir: Directorio base (exports/fp32)
        model_name: Nombre del modelo (ej: yolov11n) o None para todos
        resolution: Resolución (320, 640) o None para todas
        exclude_patterns: Patterns a excluir (ej: ["-cls", "-seg", "-pose", "-obb"])

    Returns:
        Lista de paths a archivos .onnx encontrados
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {base_dir}")

    models = []
    exclude_patterns = exclude_patterns or []

    # Si se especifica modelo, buscar solo en esa carpeta
    if model_name:
        model_dir = base_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_name}")

        # Buscar por resolución o todas
        if resolution:
            onnx_file = model_dir / f"{model_name}_{resolution}.onnx"
            if onnx_file.exists():
                models.append(onnx_file)
        else:
            models.extend(sorted(model_dir.glob("*.onnx")))

    else:
        # Buscar todos los modelos
        for model_dir in sorted(base_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            # Excluir patterns no deseados
            if any(pattern in model_dir.name for pattern in exclude_patterns):
                continue

            if resolution:
                onnx_file = model_dir / f"{model_dir.name}_{resolution}.onnx"
                if onnx_file.exists():
                    models.append(onnx_file)
            else:
                models.extend(sorted(model_dir.glob("*.onnx")))

    if not models:
        raise ValueError(
            f"No se encontraron modelos ONNX en {base_dir} "
            f"(modelo={model_name}, resolución={resolution})"
        )

    return models


class CalibrationDataLoader:
    """
    DataLoader para calibración NNCF.

    Carga frames pre-procesados (.npy) para calibración.
    """
    def __init__(self, calibration_dir: Path, resolution: int):
        """
        Args:
            calibration_dir: Directorio con frames pre-procesados
            resolution: Resolución del modelo (320 o 640)
        """
        self.calibration_dir = calibration_dir / f"preprocessed_{resolution}"

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


def calibrate_model(
    onnx_path: Path,
    calibration_data_dir: Path,
    output_dir: Path
) -> Path:
    """
    Calibra modelo ONNX a INT8 usando NNCF.

    Args:
        onnx_path: Ruta al modelo ONNX (FP32)
        calibration_data_dir: Directorio con frames de calibración
        output_dir: Directorio de salida para modelo INT8 calibrado

    Returns:
        Path al directorio del modelo OpenVINO IR (INT8 calibrado)
    """
    # Fail Fast
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX no encontrado: {onnx_path}")

    # Detectar resolución del nombre del modelo
    model_name = onnx_path.stem
    resolution = None
    if "_320" in model_name:
        resolution = 320
    elif "_640" in model_name:
        resolution = 640
    else:
        raise ValueError(f"No se pudo detectar resolución en {model_name}")

    print(f"\n🔧 Calibrando: {model_name}")
    print("-" * 60)
    print(f"   ONNX: {onnx_path}")
    print(f"   Resolución: {resolution}x{resolution}")

    # OpenVINO Core
    core = ov.Core()

    # 1. Leer modelo ONNX
    print(f"   1️⃣  Leyendo modelo ONNX...")
    model = core.read_model(onnx_path)

    # 2. Preparar calibration dataloader
    print(f"   2️⃣  Preparando calibration dataset...")
    calibration_loader = CalibrationDataLoader(calibration_data_dir, resolution)

    # 3. Crear calibration dataset para NNCF
    calibration_dataset = nncf.Dataset(calibration_loader)

    # 4. Cuantizar con NNCF (Post-Training Quantization)
    print(f"   3️⃣  Cuantizando a INT8 con NNCF...")
    print(f"      (Esto puede tomar 1-2 minutos...)")

    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        preset=nncf.QuantizationPreset.MIXED,  # MIXED = balance accuracy/speed
        # Otras opciones:
        # - PERFORMANCE: más agresivo, más rápido, menos accuracy
        # - ACCURACY: más conservador, mejor accuracy, menos speedup
    )

    # 5. Guardar modelo cuantizado
    output_dir.mkdir(parents=True, exist_ok=True)

    model_output_name = f"{model_name}_int8_calibrated"
    output_path = output_dir / model_output_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"   4️⃣  Guardando modelo INT8 calibrado...")
    ov.save_model(
        quantized_model,
        output_path / f"{model_output_name}.xml"
    )

    print(f"   ✅ Calibrado: {output_path.name}/")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Calibra modelos ONNX a INT8 usando NNCF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Calibrar todos los modelos detection (sin cls, seg, pose, obb)
  uv run calibrate_int8.py

  # Calibrar solo yolov11n
  uv run calibrate_int8.py --model yolov11n

  # Calibrar solo resolución 320
  uv run calibrate_int8.py --resolution 320

  # Calibrar modelo + resolución específica
  uv run calibrate_int8.py --model yolov11n --resolution 640

  # Incluir todos los tipos de modelos (cls, seg, pose, obb)
  uv run calibrate_int8.py --include-all
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nombre del modelo (ej: yolov11n, yolov11s). Si no se especifica, calibra todos."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[320, 640],
        default=None,
        help="Resolución del modelo (320 o 640). Si no se especifica, calibra todas."
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        default="calibration_data",
        help="Directorio con frames de calibración"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exports/int8_calibrated",
        help="Directorio de salida para modelos calibrados"
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Incluir modelos -cls, -seg, -pose, -obb (default: solo detection)"
    )

    args = parser.parse_args()

    # Configuración
    ONNX_DIR = Path("exports/fp32")
    CALIBRATION_DATA_DIR = Path(args.calibration_data)
    OUTPUT_DIR = Path(args.output)

    # Patterns a excluir (solo detection models por default)
    exclude_patterns = [] if args.include_all else ["-cls", "-seg", "-pose", "-obb"]

    print("🎯 Bakery - INT8 Calibration with NNCF")
    print("=" * 70)
    print(f"📂 Modelos ONNX: {ONNX_DIR}")
    print(f"📊 Calibration data: {CALIBRATION_DATA_DIR}")
    print(f"💾 Output: {OUTPUT_DIR}")
    if args.model:
        print(f"🎯 Modelo: {args.model}")
    if args.resolution:
        print(f"📐 Resolución: {args.resolution}")
    if exclude_patterns:
        print(f"🚫 Excluyendo: {exclude_patterns}")

    # Validar calibration data
    if not CALIBRATION_DATA_DIR.exists():
        print(f"\n❌ Error: Calibration data no encontrado en {CALIBRATION_DATA_DIR}")
        print("\n💡 Ejecuta primero:")
        print("   uv run extract_calibration_frames.py")
        return

    # Descubrir modelos
    try:
        MODELS = discover_onnx_models(
            ONNX_DIR,
            args.model,
            args.resolution,
            exclude_patterns
        )
        print(f"\n📦 Modelos ONNX encontrados: {len(MODELS)}")
        for model in MODELS:
            print(f"   - {model.parent.name}/{model.name}")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        return

    print("\n" + "=" * 70)
    print(f"🔄 Calibrando {len(MODELS)} modelos...")
    print("=" * 70)

    # Calibrar cada modelo
    calibrated_models = []
    failed_models = []

    for idx, model_path in enumerate(MODELS, 1):
        try:
            print(f"\n[{idx}/{len(MODELS)}]")
            output_path = calibrate_model(
                model_path,
                CALIBRATION_DATA_DIR,
                OUTPUT_DIR / model_path.parent.name
            )
            calibrated_models.append(output_path)

        except Exception as e:
            print(f"   ❌ Error calibrando {model_path.name}: {e}")
            failed_models.append(model_path.name)
            import traceback
            traceback.print_exc()
            continue

    # Resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE CALIBRACIÓN")
    print("=" * 70)
    print(f"✅ Modelos calibrados exitosamente: {len(calibrated_models)}")
    print(f"❌ Modelos con errores: {len(failed_models)}")

    if calibrated_models:
        print(f"\n💾 Modelos calibrados guardados en:")
        print(f"   {OUTPUT_DIR}/")
        for model_path in calibrated_models[:5]:  # Mostrar primeros 5
            print(f"   - {model_path.parent.name}/{model_path.name}/")
        if len(calibrated_models) > 5:
            print(f"   ... y {len(calibrated_models) - 5} más")

    if failed_models:
        print(f"\n⚠️  Modelos que fallaron:")
        for model_name in failed_models:
            print(f"   - {model_name}")

    print("\n" + "=" * 70)
    print("🎉 Calibración completada!")
    print("\n💡 Siguiente paso:")
    print("   # Probar INT8 calibrado en CPU con VNNI")
    print("   uv run inference_int8.py --precision int8 --device CPU \\")
    print("       --model yolov11n --resolution 320 \\")
    print("       --calibrated")


if __name__ == "__main__":
    main()
