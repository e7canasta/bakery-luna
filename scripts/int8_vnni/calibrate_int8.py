#!/usr/bin/env python3
"""
Bakery - INT8 Calibration with NNCF
====================================

Cuantiza modelos ONNX a INT8 usando NNCF (Neural Network Compression Framework)
con un dataset de calibración real para preservar accuracy.

Flujo de trabajo:
    1. Extraer frames de calibración (extract_calibration_frames.py)
    2. Calibrar modelos con este script
    3. Usar modelos calibrados en run_luna.py

Uso:
    # Calibrar modelo específico (busca ONNX en directorio raíz)
    uv run calibrate_int8.py --model yolo11n-seg --resolution 320

    # Calibrar con ONNX específico
    uv run calibrate_int8.py --onnx yolo11n-seg_320.onnx

    # Calibrar todos los ONNX disponibles
    uv run calibrate_int8.py --all --resolution 320

Referencias:
    - NNCF: https://github.com/openvinotoolkit/nncf
    - Spec: .docs/next/int8_calibration/INT8_CALIBRATION_SPEC.md
"""

from pathlib import Path
import numpy as np
import argparse
from typing import List, Iterator, Optional
import openvino as ov
import nncf


class CalibrationDataLoader:
    """
    DataLoader para calibración NNCF.

    Carga frames pre-procesados (.npy) para calibración.
    Los frames deben ser generados con extract_calibration_frames.py
    """

    def __init__(self, calibration_dir: Path, resolution: int):
        """
        Args:
            calibration_dir: Directorio base de calibración
            resolution: Resolución del modelo (320 o 640)
        """
        self.calibration_dir = calibration_dir / f"preprocessed_{resolution}"

        if not self.calibration_dir.exists():
            raise FileNotFoundError(
                f"Calibration data no encontrado: {self.calibration_dir}\n"
                f"Ejecuta primero: uv run extract_calibration_frames.py --resolution {resolution}"
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


def discover_onnx_models(
    search_dirs: List[Path],
    model_name: Optional[str] = None,
    resolution: Optional[int] = None,
    model_type: Optional[str] = None
) -> List[Path]:
    """
    Descubre modelos ONNX disponibles en múltiples directorios.

    Args:
        search_dirs: Lista de directorios donde buscar
        model_name: Nombre del modelo (ej: yolo11n-seg) o None para todos
        resolution: Resolución (320, 640) o None para todas
        model_type: Tipo de modelo (seg, pose, detection) o None para todos

    Returns:
        Lista de paths a archivos .onnx encontrados
    """
    models = []

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Buscar recursivamente
        for onnx_path in search_dir.rglob("*.onnx"):
            name = onnx_path.stem.lower()

            # Filtrar por nombre de modelo
            if model_name and model_name.lower() not in name:
                continue

            # Filtrar por resolución
            if resolution:
                if f"_{resolution}" not in name and f"_{resolution}." not in str(onnx_path):
                    continue

            # Filtrar por tipo
            if model_type:
                if model_type == "seg" and "-seg" not in name:
                    continue
                elif model_type == "pose" and "-pose" not in name:
                    continue
                elif model_type == "detection" and ("-seg" in name or "-pose" in name):
                    continue

            models.append(onnx_path)

    return sorted(set(models))  # Eliminar duplicados


def calibrate_model(
    onnx_path: Path,
    calibration_data_dir: Path,
    output_dir: Path,
    preset: str = "mixed"
) -> Path:
    """
    Calibra modelo ONNX a INT8 usando NNCF.

    Args:
        onnx_path: Ruta al modelo ONNX (FP32)
        calibration_data_dir: Directorio con frames de calibración
        output_dir: Directorio de salida para modelo INT8 calibrado
        preset: Preset de cuantización (performance, mixed, accuracy)

    Returns:
        Path al directorio del modelo OpenVINO IR (INT8 calibrado)
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX no encontrado: {onnx_path}")

    # Detectar resolución del nombre del modelo
    model_name = onnx_path.stem
    resolution = None

    for res in [192, 256, 320, 480, 640]:
        if f"_{res}" in model_name:
            resolution = res
            break

    if resolution is None:
        raise ValueError(f"No se pudo detectar resolución en {model_name}")

    print(f"\n🔧 Calibrando: {model_name}")
    print("-" * 60)
    print(f"   ONNX: {onnx_path}")
    print(f"   Resolución: {resolution}x{resolution}")
    print(f"   Preset: {preset}")

    # OpenVINO Core
    core = ov.Core()

    # 1. Leer modelo ONNX
    print(f"   1️⃣  Leyendo modelo ONNX...")
    model = core.read_model(str(onnx_path))

    # 2. Preparar calibration dataloader
    print(f"   2️⃣  Preparando calibration dataset...")
    calibration_loader = CalibrationDataLoader(calibration_data_dir, resolution)

    # 3. Crear calibration dataset para NNCF
    calibration_dataset = nncf.Dataset(calibration_loader)

    # 4. Seleccionar preset
    preset_map = {
        "performance": nncf.QuantizationPreset.PERFORMANCE,
        "mixed": nncf.QuantizationPreset.MIXED,
    }
    nncf_preset = preset_map.get(preset, nncf.QuantizationPreset.MIXED)

    # 5. Cuantizar con NNCF (Post-Training Quantization)
    print(f"   3️⃣  Cuantizando a INT8 con NNCF...")
    print(f"      (Esto puede tomar 1-2 minutos...)")

    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        preset=nncf_preset,
        subset_size=min(300, len(calibration_loader)),  # Usar hasta 300 samples
    )

    # 6. Determinar subdirectorio según tipo de modelo
    if "-seg" in model_name:
        type_dir = "segmentation"
    elif "-pose" in model_name:
        type_dir = "pose"
    else:
        type_dir = "detection"

    # 7. Guardar modelo cuantizado
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extraer nombre base del modelo (sin resolución)
    base_name = model_name.split("_")[0]  # yolo11n-seg

    model_output_name = f"{model_name}_int8_calibrated"
    output_path = output_dir / type_dir / base_name / model_output_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"   4️⃣  Guardando modelo INT8 calibrado...")
    xml_path = output_path / f"{model_output_name}.xml"
    ov.save_model(quantized_model, str(xml_path))

    print(f"   ✅ Calibrado: {xml_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Calibra modelos ONNX a INT8 usando NNCF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Calibrar modelo específico
  uv run calibrate_int8.py --model yolo11n-seg --resolution 320

  # Calibrar con ONNX específico
  uv run calibrate_int8.py --onnx yolo11n-seg_320.onnx

  # Calibrar todos los modelos de segmentación
  uv run calibrate_int8.py --type seg --resolution 320

  # Usar preset de máximo rendimiento
  uv run calibrate_int8.py --model yolo11n-seg --resolution 320 --preset performance

Flujo de trabajo:
  1. Extraer frames: uv run extract_calibration_frames.py --source video.mp4 --resolution 320
  2. Calibrar: uv run calibrate_int8.py --model yolo11n-seg --resolution 320
  3. Usar: uv run run_luna.py --seg-model exports/int8_calibrated/.../model.xml --seg-device CPU
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nombre del modelo (ej: yolo11n-seg). Busca ONNX en directorio raíz y exports/"
    )

    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Path directo al archivo ONNX (alternativa a --model)"
    )

    parser.add_argument(
        "--resolution",
        type=int,
        choices=[192, 256, 320, 480, 640],
        default=None,
        help="Resolución del modelo"
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["seg", "pose", "detection"],
        default=None,
        help="Tipo de modelo a calibrar"
    )

    parser.add_argument(
        "--calibration-data",
        type=Path,
        default=Path("calibration_data"),
        help="Directorio con frames de calibración (default: calibration_data/)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exports/int8_calibrated"),
        help="Directorio de salida (default: exports/int8_calibrated/)"
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=["performance", "mixed"],
        default="mixed",
        help="Preset de cuantización: performance (más rápido) o mixed (balance)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Calibrar todos los ONNX encontrados"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🎯 Bakery - INT8 Calibration with NNCF")
    print("=" * 70)

    # Validar argumentos
    if not args.onnx and not args.model and not args.all:
        print("❌ Debes especificar --model, --onnx, o --all")
        parser.print_help()
        return 1

    # Validar calibration data
    if not args.calibration_data.exists():
        print(f"\n❌ Calibration data no encontrado: {args.calibration_data}")
        print("\n💡 Ejecuta primero:")
        print("   uv run extract_calibration_frames.py --source VIDEO.mp4 --resolution 320")
        return 1

    # Mostrar configuración
    print(f"\n📊 Calibration data: {args.calibration_data}")
    print(f"💾 Output: {args.output}")
    print(f"⚙️  Preset: {args.preset}")

    # Recopilar modelos ONNX a calibrar
    onnx_models = []

    if args.onnx:
        # Path directo
        if not args.onnx.exists():
            print(f"❌ ONNX no encontrado: {args.onnx}")
            return 1
        onnx_models.append(args.onnx)

    else:
        # Buscar en múltiples ubicaciones
        search_dirs = [
            Path("."),  # Directorio raíz
            Path("exports/fp32"),
            Path("exports/onnx"),
        ]

        onnx_models = discover_onnx_models(
            search_dirs,
            model_name=args.model,
            resolution=args.resolution,
            model_type=args.type
        )

    if not onnx_models:
        print("\n❌ No se encontraron modelos ONNX")
        print("\n💡 Asegúrate de tener archivos .onnx en el directorio")
        print("   Puedes generarlos con: uv run export_int8.py --model yolo11n-seg --resolution 320")
        return 1

    print(f"\n📦 Modelos ONNX a calibrar: {len(onnx_models)}")
    for model in onnx_models:
        print(f"   - {model}")

    print("\n" + "=" * 70)

    # Calibrar cada modelo
    calibrated = []
    failed = []

    for onnx_path in onnx_models:
        try:
            output_path = calibrate_model(
                onnx_path,
                args.calibration_data,
                args.output,
                args.preset
            )
            calibrated.append(output_path)
        except Exception as e:
            print(f"\n❌ Error calibrando {onnx_path.name}: {e}")
            failed.append(onnx_path)

    # Resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)

    if calibrated:
        print(f"\n✅ Calibrados exitosamente: {len(calibrated)}")
        for path in calibrated:
            print(f"   - {path}")

    if failed:
        print(f"\n❌ Fallidos: {len(failed)}")
        for path in failed:
            print(f"   - {path}")

    if calibrated:
        print("\n💡 Próximo paso:")
        print("   uv run run_luna.py \\")
        print(f"       --seg-model {calibrated[0]}/...xml \\")
        print("       --seg-device CPU")

    print("\n" + "=" * 70)

    return 0 if not failed else 1


if __name__ == "__main__":
    exit(main())
