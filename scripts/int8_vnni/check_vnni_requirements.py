#!/usr/bin/env python3
"""
Verifica que todos los paquetes necesarios para VNNI/INT8 estén instalados.

Uso:
    uv run scripts/int8_vnni/check_vnni_requirements.py
"""

import sys

def check_package(package_name, import_name=None):
    """Verifica si un paquete está instalado."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def main():
    print("🔍 Verificando paquetes necesarios para VNNI/INT8...\n")
    
    required_packages = [
        ("openvino", "openvino"),
        ("nncf", "nncf"),  # Para calibración INT8 (opcional pero recomendado)
        ("psutil", "psutil"),  # Para monitoreo de sistema
    ]
    
    all_ok = True
    
    for package_name, import_name in required_packages:
        installed, error = check_package(package_name, import_name)
        
        if installed:
            print(f"✅ {package_name}: Instalado")
        else:
            print(f"❌ {package_name}: NO instalado")
            print(f"   Error: {error}")
            all_ok = False
    
    print("\n" + "=" * 70)
    
    if all_ok:
        print("✅ Todos los paquetes necesarios están instalados")
        print("\n💡 Nota: VNNI es una característica del CPU, no requiere paquetes adicionales.")
        print("   OpenVINO detecta y usa VNNI automáticamente si tu CPU lo tiene.")
    else:
        print("❌ Faltan algunos paquetes")
        print("\n💡 Para instalar:")
        print("   uv sync")
        print("   # O específicamente:")
        for package_name, _ in required_packages:
            if not check_package(package_name)[0]:
                print(f"   uv pip install {package_name}")
    
    print("\n" + "=" * 70)
    
    # Verificar OpenVINO específicamente
    try:
        import openvino as ov
        core = ov.Core()
        print("\n🚀 OpenVINO Info:")
        print(f"   Versión: {ov.__version__}")
        print(f"   Devices disponibles: {core.available_devices}")
        
        # Verificar CPU capabilities
        try:
            cpu_caps = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
            print(f"   CPU capabilities: {cpu_caps}")
            
            if isinstance(cpu_caps, list):
                has_int8 = any("INT8" in str(c).upper() for c in cpu_caps)
            else:
                has_int8 = "INT8" in str(cpu_caps).upper()
            
            if has_int8:
                print("   ✅ INT8 soportado en CPU")
            else:
                print("   ⚠️  INT8 no reportado (puede estar disponible)")
                
        except Exception as e:
            print(f"   ⚠️  No se pudo obtener CPU capabilities: {e}")
            
    except ImportError:
        print("\n❌ OpenVINO no está instalado")
        print("   Instalar con: uv pip install openvino openvino-dev")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
