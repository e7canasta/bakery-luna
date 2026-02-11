#!/usr/bin/env python3
"""
Verifica soporte de VNNI (Vector Neural Network Instructions) en CPU.

VNNI = Aceleración hardware de INT8 en CPU Intel (desde Ice Lake).
Sin VNNI, INT8 se ejecuta en software (más lento).

Uso:
    uv run verify_vnni.py
"""

import subprocess
import platform
from pathlib import Path


def check_vnni_cpuinfo():
    """
    Verifica VNNI en /proc/cpuinfo (Linux).

    VNNI flags a buscar:
    - avx512_vnni: AVX-512 VNNI (Ice Lake+)
    - avx_vnni: AVX2 VNNI (Alder Lake+)
    - avx512_bf16: BFloat16 support (también indica VNNI capability)
    """
    if platform.system() != "Linux":
        print("⚠️  /proc/cpuinfo solo disponible en Linux")
        return None

    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.exists():
        print("❌ /proc/cpuinfo no encontrado")
        return None

    with open(cpuinfo_path) as f:
        cpuinfo = f.read()

    # Buscar flags relevantes
    vnni_flags = {
        "avx512_vnni": False,
        "avx_vnni": False,
        "avx512_bf16": False,
    }

    for line in cpuinfo.split("\n"):
        if line.startswith("flags"):
            flags = line.split(":")[1].strip().split()
            for flag in vnni_flags:
                if flag in flags:
                    vnni_flags[flag] = True

    return vnni_flags


def check_vnni_lscpu():
    """
    Verifica VNNI usando lscpu (más limpio).

    lscpu muestra CPU flags de forma más legible.
    """
    try:
        result = subprocess.run(
            ["lscpu"],
            capture_output=True,
            text=True,
            check=True
        )

        output = result.stdout.lower()

        # Buscar flags en output de lscpu
        vnni_detected = False
        if "avx512_vnni" in output or "avx_vnni" in output:
            vnni_detected = True

        return vnni_detected, result.stdout

    except FileNotFoundError:
        print("⚠️  lscpu no disponible")
        return None, None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando lscpu: {e}")
        return None, None


def check_openvino_cpu_capabilities():
    """
    Verifica qué optimizaciones CPU reporta OpenVINO.

    OpenVINO detecta automáticamente CPU features y las reporta
    en OPTIMIZATION_CAPABILITIES.
    """
    try:
        import openvino as ov

        core = ov.Core()

        # Obtener capabilities del device CPU
        cpu_caps = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")

        # Obtener nombre completo del device
        cpu_name = core.get_property("CPU", "FULL_DEVICE_NAME")

        return cpu_name, cpu_caps

    except ImportError:
        print("❌ OpenVINO no instalado (pip install openvino)")
        return None, None
    except Exception as e:
        print(f"❌ Error consultando OpenVINO: {e}")
        return None, None


def get_cpu_model():
    """
    Obtiene modelo del CPU.
    """
    try:
        result = subprocess.run(
            ["lscpu"],
            capture_output=True,
            text=True,
            check=True
        )

        for line in result.stdout.split("\n"):
            if line.startswith("Model name:"):
                return line.split(":")[1].strip()

        return "Unknown"

    except (FileNotFoundError, subprocess.CalledProcessError):
        return "Unknown"


def interpret_results(vnni_flags, vnni_lscpu, cpu_name, cpu_caps):
    """
    Interpreta resultados y da veredicto final.
    """
    print("\n" + "=" * 70)
    print("📊 ANÁLISIS DE SOPORTE VNNI")
    print("=" * 70)

    cpu_model = get_cpu_model()
    print(f"\n🖥️  CPU: {cpu_model}")

    # Resultado de /proc/cpuinfo
    if vnni_flags:
        print("\n📋 Flags en /proc/cpuinfo:")
        for flag, present in vnni_flags.items():
            status = "✅" if present else "❌"
            print(f"   {status} {flag}")

    # Resultado de lscpu
    if vnni_lscpu is not None:
        status = "✅" if vnni_lscpu else "❌"
        print(f"\n🔍 lscpu detection: {status}")

    # Resultado de OpenVINO
    if cpu_name and cpu_caps:
        print(f"\n🚀 OpenVINO CPU detection:")
        print(f"   Device: {cpu_name}")
        print(f"   Capabilities: {cpu_caps}")

        # Analizar capabilities
        # cpu_caps ya es una lista desde OpenVINO
        caps_list = cpu_caps if isinstance(cpu_caps, list) else [c.strip() for c in cpu_caps.split(",")]

        vnni_in_openvino = any("VNNI" in str(c).upper() for c in caps_list)
        int8_in_openvino = any("INT8" in str(c).upper() for c in caps_list)

        # Nota: OpenVINO puede no reportar "VNNI" explícitamente, pero si tiene INT8
        # y el CPU tiene VNNI (detectado en cpuinfo/lscpu), OpenVINO lo usará automáticamente
        print(f"\n   📌 VNNI explícito en capabilities: {'✅ Sí' if vnni_in_openvino else '❌ No (pero puede estar disponible)'}")
        print(f"   📌 INT8 soportado: {'✅ Sí' if int8_in_openvino else '❌ No'}")
        if int8_in_openvino and not vnni_in_openvino:
            print(f"   ℹ️  Nota: OpenVINO no reporta 'VNNI' explícitamente, pero si tu CPU")
            print(f"      tiene VNNI (verificado arriba), OpenVINO lo usará automáticamente para INT8")

    # Veredicto final
    print("\n" + "=" * 70)
    print("🎯 VEREDICTO FINAL")
    print("=" * 70)

    has_vnni = False
    if vnni_flags and any(vnni_flags.values()):
        has_vnni = True
    if vnni_lscpu:
        has_vnni = True

    if has_vnni:
        print("✅ VNNI HABILITADO")
        print("\n   Tu CPU tiene aceleración hardware de INT8.")
        print("   INT8 inference en CPU será ~2-4x más rápido que FP32.")
        print("\n   💡 Recomendación: Usar INT8 en CPU para mejor performance.")
    else:
        print("❌ VNNI NO DETECTADO")
        print("\n   Tu CPU no tiene aceleración hardware de INT8,")
        print("   o es un modelo anterior a Ice Lake (2019).")
        print("\n   ⚠️  INT8 se ejecutará en software (más lento).")
        print("   💡 Recomendación: Usar FP32/FP16, o actualizar CPU.")

    # Guía de generaciones Intel con VNNI
    print("\n" + "=" * 70)
    print("📚 REFERENCIA: Generaciones Intel con VNNI")
    print("=" * 70)
    print("""
Desktop:
   ✅ Ice Lake (10th gen, 2019)      → AVX-512 VNNI
   ✅ Alder Lake (12th gen, 2021)    → AVX2 VNNI
   ✅ Raptor Lake (13th gen, 2022)   → AVX2 VNNI
   ❌ Coffee Lake (8th/9th gen)      → Sin VNNI

Laptop:
   ✅ Ice Lake (10th gen, 2019)      → AVX-512 VNNI
   ✅ Tiger Lake (11th gen, 2020)    → AVX-512 VNNI
   ✅ Alder Lake (12th gen, 2021)    → AVX2 VNNI
   ❌ Kaby Lake (7th gen)            → Sin VNNI
    """)


def main():
    print("🔍 Verificando soporte de VNNI en CPU...\n")

    # Check 1: /proc/cpuinfo
    print("1️⃣  Checkeando /proc/cpuinfo...")
    vnni_flags = check_vnni_cpuinfo()

    # Check 2: lscpu
    print("2️⃣  Checkeando lscpu...")
    vnni_lscpu, lscpu_output = check_vnni_lscpu()

    # Check 3: OpenVINO
    print("3️⃣  Checkeando OpenVINO capabilities...")
    cpu_name, cpu_caps = check_openvino_cpu_capabilities()

    # Interpretar resultados
    interpret_results(vnni_flags, vnni_lscpu, cpu_name, cpu_caps)


if __name__ == "__main__":
    main()
