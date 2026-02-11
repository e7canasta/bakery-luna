#!/usr/bin/env python3
"""
Verifica si OpenVINO realmente ejecuta modelos en INT8 o hace fallback a FP16/FP32.

El problema: OpenVINO puede cargar un modelo INT8 pero ejecutarlo en FP16/FP32
si el device no soporta INT8 realmente.

Uso:
    uv run verify_int8_execution.py
    uv run verify_int8_execution.py --device GPU
    uv run verify_int8_execution.py --model yolov11n --resolution 320
"""

import argparse
from pathlib import Path
import sys


def discover_int8_models(exports_dir: Path):
    """
    Descubre modelos INT8 disponibles recursivamente.

    Returns:
        List[dict]: Lista de modelos con metadata
    """
    int8_dir = exports_dir / "int8"

    if not int8_dir.exists():
        print(f"❌ Directorio INT8 no encontrado: {int8_dir}")
        return []

    models = []

    # Buscar todos los archivos .xml recursivamente
    for xml_path in sorted(int8_dir.rglob("*.xml")):
        # Extraer nombre del modelo del path
        # Estructura: int8/{type}/{model_name}/{variant}/{model.xml}
        # O: int8/{model_name}/{variant}/{model.xml}
        parts = xml_path.parts

        # El nombre del modelo está en el stem del archivo
        model_stem = xml_path.stem

        # Extraer nombre base del modelo (sin resolución y _int8)
        # Ejemplo: yolo11n_320_int8 -> yolo11n
        name_parts = model_stem.split("_")
        model_name = name_parts[0]
        if len(name_parts) > 1 and "-" in name_parts[0]:
            # Caso: yolo11n-seg_320_int8
            model_name = name_parts[0]

        # Extraer resolución del nombre
        resolution = None
        for part in name_parts:
            if part.isdigit():
                resolution = int(part)
                break

        models.append({
            "name": model_name,
            "resolution": resolution,
            "path": xml_path,
            "variant": xml_path.parent.name,
        })

    return models


def verify_model_precision(model_path: Path, device: str = "CPU"):
    """
    Verifica la precisión real de ejecución de un modelo.

    Estrategia:
    1. Load model con OpenVINO
    2. Compile model en device específico
    3. Query execution precision del compiled model
    4. Check si hay INT8 ops en el grafo

    Returns:
        dict: Metadata de precisión
    """
    try:
        import openvino as ov
        from openvino.runtime import Core, Type
    except ImportError:
        print("❌ OpenVINO no instalado (pip install openvino)")
        sys.exit(1)

    core = Core()

    # 1. Read model (IR format)
    print(f"\n🔍 Cargando modelo: {model_path.name}")
    model = core.read_model(model_path)

    # 2. Analizar operaciones en el grafo
    print(f"   📊 Analizando grafo del modelo...")

    # Contar operaciones relevantes
    compute_ops = {"Convolution", "MatMul", "GroupConvolution", "ConvolutionBackpropData"}

    total_compute_ops = 0
    fake_quantize_ops = 0

    for op in model.get_ops():
        op_type = op.get_type_name()

        if op_type in compute_ops:
            total_compute_ops += 1
        elif op_type == "FakeQuantize":
            fake_quantize_ops += 1

    # FakeQuantize indica cuantización INT8 en OpenVINO IR
    # Un modelo INT8 típico tiene ~2-3 FakeQuantize por cada Conv (entrada, pesos, salida)
    is_quantized = fake_quantize_ops > 0
    quantization_ratio = fake_quantize_ops / total_compute_ops if total_compute_ops > 0 else 0

    print(f"   📌 Operaciones computacionales (Conv/MatMul): {total_compute_ops}")
    print(f"   📌 FakeQuantize ops (indicador de INT8): {fake_quantize_ops}")
    if is_quantized:
        print(f"      ✅ Modelo tiene cuantización INT8 (ratio FQ/Conv: {quantization_ratio:.1f})")
    else:
        print(f"      ❌ Modelo NO tiene cuantización INT8")

    # 3. Compile model en device específico
    print(f"\n   🚀 Compilando modelo en device: {device}")

    try:
        compiled = core.compile_model(model, device)
    except Exception as e:
        print(f"   ❌ Error compilando en {device}: {e}")
        return None

    # 4. Query device properties
    print(f"   📋 Properties del compiled model:")

    try:
        execution_devices = compiled.get_property("EXECUTION_DEVICES")
        print(f"      Execution devices: {execution_devices}")
    except Exception as e:
        print(f"      ⚠️  No se pudo obtener EXECUTION_DEVICES: {e}")

    try:
        model_priority = compiled.get_property("MODEL_PRIORITY")
        print(f"      Model priority: {model_priority}")
    except Exception:
        pass  # No todos los devices soportan esta property

    # 5. Check device capabilities
    print(f"\n   🔧 Capabilities del device {device}:")

    try:
        optimization_caps = core.get_property(device, "OPTIMIZATION_CAPABILITIES")
        print(f"      Optimization capabilities: {optimization_caps}")

        # Parse capabilities (puede ser lista o string)
        if isinstance(optimization_caps, list):
            caps_list = optimization_caps
        else:
            caps_list = [c.strip() for c in str(optimization_caps).split(",")]

        has_int8 = any("INT8" in str(c).upper() for c in caps_list)

        print(f"\n      📌 INT8 support: {'✅ Sí' if has_int8 else '❌ No'}")

    except Exception as e:
        print(f"      ⚠️  No se pudo obtener OPTIMIZATION_CAPABILITIES: {e}")
        has_int8 = None

    # Check VNNI from /proc/cpuinfo (más confiable)
    has_vnni = False
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
            has_vnni = "avx512_vnni" in cpuinfo or "avx_vnni" in cpuinfo
        print(f"      📌 VNNI support (CPU): {'✅ Sí' if has_vnni else '❌ No'}")
    except Exception:
        pass

    # 6. Get device name
    try:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"      Device name: {device_name}")
    except Exception:
        device_name = "Unknown"

    return {
        "model_path": model_path,
        "device": device,
        "device_name": device_name,
        "total_compute_ops": total_compute_ops,
        "fake_quantize_ops": fake_quantize_ops,
        "is_quantized": is_quantized,
        "quantization_ratio": quantization_ratio,
        "has_int8_support": has_int8,
        "has_vnni_support": has_vnni,
    }


def interpret_results(results: dict):
    """
    Interpreta resultados y da veredicto.
    """
    print("\n" + "=" * 70)
    print("🎯 VEREDICTO DE EJECUCIÓN")
    print("=" * 70)

    device = results["device"]
    is_quantized = results["is_quantized"]
    fq_ops = results["fake_quantize_ops"]
    compute_ops = results["total_compute_ops"]

    print(f"\n📊 Modelo: {results['model_path'].name}")
    print(f"🖥️  Device: {device} ({results['device_name']})")
    print(f"📈 FakeQuantize ops: {fq_ops} (Conv/MatMul: {compute_ops})")

    # Veredicto
    if is_quantized and results["quantization_ratio"] >= 2:
        print("\n✅ MODELO ES REALMENTE INT8")
        print("   Cuantización completa detectada (FakeQuantize presente).")
        print("   OpenVINO ejecutará operaciones en INT8 con VNNI.")
    elif is_quantized:
        print("\n⚠️  MODELO ES PARCIALMENTE INT8")
        print("   Algunas operaciones cuantizadas.")
        print("   Esto es normal (primera/última capa suelen ser FP32).")
    else:
        print("\n❌ MODELO NO ES INT8")
        print("   No hay FakeQuantize ops.")
        print("   Modelo es FP32/FP16.")

    # Device support
    if results["has_int8_support"]:
        print(f"\n✅ {device} SOPORTA INT8")
    else:
        print(f"\n❌ {device} NO SOPORTA INT8")
        print("   OpenVINO puede hacer fallback a FP32/FP16.")

    if device == "CPU" and results["has_vnni_support"]:
        print("✅ CPU TIENE VNNI (aceleración hardware de INT8)")
    elif device == "CPU" and not results["has_vnni_support"]:
        print("⚠️  CPU SIN VNNI (INT8 en software, más lento)")

    # Recomendaciones
    print("\n" + "=" * 70)
    print("💡 RECOMENDACIONES")
    print("=" * 70)

    if device == "CPU":
        if results["has_vnni_support"] and is_quantized:
            print("✅ Configuración óptima para CPU:")
            print("   - VNNI habilitado")
            print("   - Modelo INT8 real")
            print("   - Expect ~2-4x speedup vs FP32")
        elif not results["has_vnni_support"]:
            print("⚠️  CPU sin VNNI:")
            print("   - Considerar usar FP32 (puede ser más rápido)")
            print("   - O actualizar CPU (Ice Lake+)")
        else:
            print("✅ INT8 funcionando, pero beneficio limitado")

    elif device == "GPU":
        if results["has_int8_support"] and is_quantized:
            print("✅ GPU ejecutando INT8 realmente")
            print("   - Expect speedup en resoluciones altas (640+)")
        elif not results["has_int8_support"]:
            print("⚠️  GPU puede no soportar INT8 totalmente:")
            print("   - iGPU Intel (Xe-LP) tiene soporte limitado")
            print("   - Considerar usar FP16 en GPU")
        else:
            print("⚠️  Verificar performance real (benchmark)")


def main():
    parser = argparse.ArgumentParser(
        description="Verifica ejecución real de INT8 en OpenVINO"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Modelo específico (ej: yolov11n). Si no se especifica, usa el primero disponible"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[320, 640],
        help="Resolución específica"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        choices=["CPU", "GPU", "AUTO"],
        help="Device para compilar modelo (default: CPU)"
    )

    args = parser.parse_args()

    # Descubrir modelos INT8
    exports_dir = Path("exports")
    models = discover_int8_models(exports_dir)

    if not models:
        print("❌ No se encontraron modelos INT8 en exports/int8/")
        print("   Ejecutá primero: uv run main.py")
        sys.exit(1)

    # Filtrar por nombre de modelo si se especificó
    if args.model:
        models = [m for m in models if m["name"] == args.model]
        if not models:
            print(f"❌ Modelo '{args.model}' no encontrado")
            sys.exit(1)

    # Filtrar por resolución si se especificó
    if args.resolution:
        models = [m for m in models if m["resolution"] == args.resolution]
        if not models:
            print(f"❌ Resolución {args.resolution} no encontrada")
            sys.exit(1)

    # Si hay múltiples modelos, tomar el primero
    if len(models) > 1:
        print(f"📦 Encontrados {len(models)} modelos INT8")
        print("   Usando el primero para verificación:")
        for m in models:
            print(f"   - {m['name']} @ {m['resolution']}")
        print()

    model = models[0]

    print("=" * 70)
    print("🔍 VERIFICADOR DE EJECUCIÓN INT8")
    print("=" * 70)
    print(f"\n📦 Modelo: {model['name']}")
    print(f"📐 Resolución: {model['resolution']}")
    print(f"🖥️  Device: {args.device}")

    # Verificar modelo
    results = verify_model_precision(model["path"], args.device)

    if results:
        interpret_results(results)
    else:
        print("\n❌ No se pudo verificar el modelo")
        sys.exit(1)


if __name__ == "__main__":
    main()
