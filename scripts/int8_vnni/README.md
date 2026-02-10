# Scripts INT8/VNNI

Este directorio contiene scripts para verificación, calibración y benchmarking de modelos INT8 con soporte VNNI (Vector Neural Network Instructions) en CPU Intel.

## Scripts Disponibles

### Verificación

- **`verify_vnni.py`**: Verifica soporte VNNI en CPU
  ```bash
  uv run scripts/int8_vnni/verify_vnni.py
  ```

- **`verify_int8_execution.py`**: Verifica ejecución real de INT8 (no fallback)
  ```bash
  uv run scripts/int8_vnni/verify_int8_execution.py --device CPU
  ```

- **`verify_load_balancing.py`**: Monitorea distribución de carga en tiempo real
  ```bash
  uv run scripts/int8_vnni/verify_load_balancing.py --seg-device GPU --pose-device CPU
  ```

### Calibración

- **`calibrate_int8.py`**: Calibra modelos ONNX a INT8 usando NNCF
  ```bash
  uv run scripts/int8_vnni/calibrate_int8.py --model yolov11n --resolution 320
  ```

### Benchmarking

- **`benchmark_cross_device.py`**: Compara INT8 CPU vs GPU vs FP16 GPU
  ```bash
  uv run scripts/int8_vnni/benchmark_cross_device.py --model yolov11n --resolution 320
  ```

- **`benchmark_cpu.py`**: Mide carga operacional del CPU
  ```bash
  uv run scripts/int8_vnni/benchmark_cpu.py --device CPU
  ```

- **`benchmark_hybrid_cpu_gpu.py`**: Benchmark pipeline dual con balanceo CPU/GPU
  ```bash
  uv run scripts/int8_vnni/benchmark_hybrid_cpu_gpu.py
  ```

## Documentación Completa

Para guía completa sobre INT8/VNNI y balanceo de carga, ver:
- [`docs/INT8_VNNI_LOAD_BALANCING.md`](../../docs/INT8_VNNI_LOAD_BALANCING.md)

## Notas

- Los scripts originales también están disponibles en la raíz del proyecto para compatibilidad
- Todos los scripts requieren OpenVINO instalado
- Para calibración INT8, se requiere NNCF (`pip install nncf`)
