# INT8/VNNI y Balanceo de Carga CPU/GPU

## Introducción

Este documento explica cómo usar **VNNI (Vector Neural Network Instructions)** para acelerar inferencia INT8 en CPU Intel, y cómo distribuir carga entre CPU y GPU en pipelines duales para optimizar performance y reducir saturación de dispositivos.

---

## ¿Qué es VNNI?

**VNNI (Vector Neural Network Instructions)** no es un chip separado, sino **instrucciones hardware del CPU Intel** que aceleran operaciones de redes neuronales en precisión INT8.

### Características Clave

- **No es un chip separado**: VNNI son instrucciones integradas en el CPU
- **Aceleración hardware**: ~2-4x speedup vs FP32 en CPU
- **Disponible desde**: Ice Lake (10th gen, 2019+)
- **Detección automática**: OpenVINO detecta VNNI automáticamente si está disponible

### Generaciones Intel con VNNI

**Desktop:**
- ✅ Ice Lake (10th gen, 2019) → AVX-512 VNNI
- ✅ Alder Lake (12th gen, 2021) → AVX2 VNNI
- ✅ Raptor Lake (13th gen, 2022) → AVX2 VNNI
- ❌ Coffee Lake (8th/9th gen) → Sin VNNI

**Laptop:**
- ✅ Ice Lake (10th gen, 2019) → AVX-512 VNNI
- ✅ Tiger Lake (11th gen, 2020) → AVX-512 VNNI
- ✅ Alder Lake (12th gen, 2021) → AVX2 VNNI
- ❌ Kaby Lake (7th gen) → Sin VNNI

---

## INT8 en CPU vs GPU

### Ventajas de INT8 en CPU (con VNNI)

1. **Aceleración hardware**: VNNI acelera operaciones INT8 nativamente
2. **Baja latencia**: CPU tiene menor latencia que GPU para inferencias pequeñas
3. **Menor consumo**: CPU con VNNI es más eficiente energéticamente
4. **Sin transferencias**: No hay overhead de transferencia CPU↔GPU
5. **Libera GPU**: Permite usar GPU para otras tareas

### Cuándo Usar CPU vs GPU

**Usar CPU (INT8/VNNI) cuando:**
- ✅ Resoluciones bajas (320x320)
- ✅ Latencia es prioritaria
- ✅ Single frame processing
- ✅ CPU tiene VNNI (Ice Lake+)
- ✅ Quieres liberar GPU para otras tareas

**Usar GPU (FP16) cuando:**
- ✅ Resoluciones altas (640x640+)
- ✅ Throughput es prioritario
- ✅ Batch processing
- ✅ GPU está disponible y no saturada

### Performance Esperada

Basado en benchmarks típicos (Intel i7-1195G7, Tiger Lake):

| Configuración | Resolución | FPS | Notas |
|--------------|------------|-----|-------|
| CPU INT8 (VNNI) | 320x320 | ~78-85 | Óptimo para edge |
| CPU FP16 | 320x320 | ~83 | Similar a INT8 sin calibración |
| GPU FP16 | 320x320 | ~20-25 | Overhead > ganancia |
| GPU FP16 | 640x640 | ~22-25 | GPU gana en resoluciones altas |
| CPU INT8 (VNNI) | 640x640 | ~15-18 | CPU limitado en resoluciones altas |

**Nota**: INT8 sin calibración puede ser más lento que FP16. Para mejor performance INT8, usar modelos calibrados con NNCF (ver `calibrate_int8.py`).

---

## Estrategias de Balanceo de Carga

### Pipeline Dual: CPU (INT8) + GPU (FP16)

En un pipeline dual (ej: segmentation + pose estimation), puedes distribuir la carga entre CPU y GPU:

**Configuración Híbrida Recomendada:**
- **Segmentation en GPU (FP16)**: Modelo más pesado, se beneficia de paralelismo GPU
- **Pose en CPU (INT8/VNNI)**: Modelo más ligero, baja latencia en CPU

**Ventajas:**
- ✅ Reduce saturación de GPU
- ✅ Libera CPU para otras tareas
- ✅ Mejor utilización de recursos
- ✅ Menor latencia total del pipeline

### Casos de Uso

1. **Producción en Edge Devices**:
   - Pose en CPU (INT8) → Baja latencia
   - Segmentation en GPU (FP16) → Mejor accuracy

2. **Servidor con Múltiples Streams**:
   - Distribuir modelos entre CPU y GPU
   - Procesar más streams simultáneamente

3. **Optimización de Recursos**:
   - Balancear carga para evitar saturación
   - Maximizar throughput total

### Configuración en OpenVINO

OpenVINO soporta diferentes devices por modelo nativamente:

```python
from bakery.adapters.openvino.inference_engine import InferenceEngine
from bakery.core.entities.model_config import ModelConfig, Device

# Segmentation en GPU
seg_config = ModelConfig(
    model_path=Path("exports/fp16/yolov11s-seg_640_fp16.xml"),
    device=Device.GPU,
    # ...
)

# Pose en CPU (INT8)
pose_config = ModelConfig(
    model_path=Path("exports/int8/yolov11n-pose_320_int8.xml"),
    device=Device.CPU,
    # ...
)

# Crear engines
seg_engine = InferenceEngine(seg_config)  # Compila en GPU
pose_engine = InferenceEngine(pose_config)  # Compila en CPU

# Pipeline dual automáticamente usa diferentes devices
pipeline = DualModelPipeline(seg_engine, pose_engine, config)
```

---

## Guía de Uso

### 1. Verificar Soporte VNNI

Antes de usar INT8 en CPU, verifica que tu CPU tenga VNNI:

```bash
uv run verify_vnni.py
# O desde scripts organizados:
uv run scripts/int8_vnni/verify_vnni.py
```

**Salida esperada:**
```
✅ VNNI HABILITADO
   Tu CPU tiene aceleración hardware de INT8.
   INT8 inference en CPU será ~2-4x más rápido que FP32.
```

### 2. Verificar Ejecución Real de INT8

Verifica que OpenVINO realmente ejecuta INT8 (no fallback a FP32/FP16):

```bash
uv run verify_int8_execution.py --device CPU
# O:
uv run scripts/int8_vnni/verify_int8_execution.py --device CPU
```

**Salida esperada:**
```
✅ MODELO ES REALMENTE INT8
   La mayoría de operaciones usan INT8.
✅ CPU TIENE VNNI (aceleración hardware de INT8)
```

### 3. Benchmark Híbrido CPU/GPU

Compara diferentes configuraciones de balanceo de carga:

```bash
uv run scripts/int8_vnni/benchmark_hybrid_cpu_gpu.py
```

**Configuraciones probadas:**
- Ambos en CPU (INT8)
- Ambos en GPU (FP16)
- Híbrido: Seg@GPU + Pose@CPU
- Híbrido: Seg@CPU + Pose@GPU

**Salida incluye:**
- FPS por configuración
- CPU% y carga operacional
- Memoria utilizada
- Recomendaciones

### 4. Monitorear Balanceo de Carga en Tiempo Real

Verifica que la carga esté distribuida correctamente durante inferencia:

```bash
uv run scripts/int8_vnni/verify_load_balancing.py \
    --seg-device GPU \
    --pose-device CPU \
    --duration 30
```

**Métricas mostradas:**
- CPU% por core
- GPU utilization (si disponible)
- Memoria
- Throughput por dispositivo
- Análisis de distribución

### 5. Calibrar Modelos INT8 (Opcional)

Para mejor performance INT8, calibra modelos con NNCF:

```bash
# Extraer frames de calibración
uv run extract_calibration_frames.py

# Calibrar modelos
uv run calibrate_int8.py --model yolov11n --resolution 320
# O:
uv run scripts/int8_vnni/calibrate_int8.py --model yolov11n
```

**Beneficios:**
- Mejor accuracy vs INT8 sin calibración
- Posible mejora de speedup
- Optimización completa del pipeline INT8

---

## Resultados y Recomendaciones

### Benchmarks Típicos

**Pipeline Dual (Segmentation + Pose):**

| Configuración | FPS | CPU% | Notas |
|--------------|-----|------|-------|
| Ambos CPU (INT8) | ~15-20 | 60-80% | CPU saturado |
| Ambos GPU (FP16) | ~18-22 | 20-30% | GPU saturado |
| Seg@GPU + Pose@CPU | ~20-25 | 40-60% | ✅ Balanceado |
| Seg@CPU + Pose@GPU | ~18-22 | 50-70% | CPU más cargado |

**Recomendación**: Seg@GPU + Pose@CPU generalmente ofrece mejor balance.

### Mejores Prácticas

1. **Verificar VNNI primero**: No usar INT8 en CPU sin VNNI (será más lento)
2. **Calibrar modelos INT8**: Para mejor accuracy y posible speedup
3. **Monitorear carga**: Usar `verify_load_balancing.py` para validar distribución
4. **Benchmark antes de producción**: Probar diferentes configuraciones
5. **Considerar resolución**: CPU gana en 320, GPU en 640+

### Troubleshooting

**Problema**: INT8 en CPU es más lento que FP16
- **Causa**: CPU sin VNNI o modelo no calibrado
- **Solución**: Verificar VNNI con `verify_vnni.py`, calibrar modelo

**Problema**: GPU no disponible
- **Causa**: Driver no instalado o GPU no compatible
- **Solución**: Usar CPU para ambos modelos, o verificar drivers

**Problema**: Carga no distribuida
- **Causa**: Un modelo mucho más pesado que el otro
- **Solución**: Balancear modelos (mover modelo ligero a CPU, pesado a GPU)

**Problema**: Bajo FPS en configuración híbrida
- **Causa**: Overhead de múltiples devices
- **Solución**: Probar configuración no-híbrida, o reducir resolución

---

## Scripts Disponibles

Todos los scripts están organizados en `scripts/int8_vnni/`:

| Script | Propósito | Uso |
|--------|-----------|-----|
| `verify_vnni.py` | Verificar soporte VNNI | `uv run scripts/int8_vnni/verify_vnni.py` |
| `verify_int8_execution.py` | Verificar ejecución INT8 real | `uv run scripts/int8_vnni/verify_int8_execution.py --device CPU` |
| `calibrate_int8.py` | Calibrar modelos INT8 con NNCF | `uv run scripts/int8_vnni/calibrate_int8.py --model yolov11n` |
| `benchmark_cross_device.py` | Benchmark CPU vs GPU | `uv run scripts/int8_vnni/benchmark_cross_device.py` |
| `benchmark_cpu.py` | Benchmark carga operacional CPU | `uv run scripts/int8_vnni/benchmark_cpu.py` |
| `benchmark_hybrid_cpu_gpu.py` | Benchmark pipeline híbrido | `uv run scripts/int8_vnni/benchmark_hybrid_cpu_gpu.py` |
| `verify_load_balancing.py` | Monitorear balanceo de carga | `uv run scripts/int8_vnni/verify_load_balancing.py` |

**Nota**: Los scripts originales también están disponibles en la raíz del proyecto para compatibilidad.

---

## Referencias

- **OpenVINO INT8 Quantization**: https://docs.openvino.ai/latest/ptq_introduction.html
- **NNCF (Neural Network Compression Framework)**: https://github.com/openvinotoolkit/nncf
- **VNNI Instructions**: Intel Architecture Instruction Set Extensions Programming Reference
- **Bakery Pipeline**: Ver `README_LUNA.md` para arquitectura completa

---

## Conclusión

Usar INT8 con VNNI en CPU Intel permite:
- ✅ Aceleración hardware de inferencia INT8
- ✅ Reducir carga en GPU
- ✅ Mejor balanceo de recursos en pipelines duales
- ✅ Menor latencia para modelos ligeros

La estrategia híbrida (CPU INT8 + GPU FP16) es especialmente útil para:
- Pipelines duales (segmentation + pose)
- Edge devices con CPU moderno
- Optimización de recursos en servidores

**Siguiente paso**: Ejecutar `benchmark_hybrid_cpu_gpu.py` para encontrar la configuración óptima en tu hardware.

---

*Última actualización: 2026-02-09 | Bakery Luna Phase 1.6*
