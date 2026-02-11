# Evaluación: INT8/VNNI y Balanceo de Carga CPU/GPU

## Resumen Ejecutivo

**Veredicto: El plan propuesto es técnicamente sólido y viable.**

La propuesta en `.docs/next/int8_vnni_cpu_load_balancing/` está bien fundamentada y alineada con las mejores prácticas de OpenVINO. La arquitectura actual de bakery-luna **ya soporta INT8** con un único gap menor.

---

## Análisis Técnico

### 1. VNNI - Validación Técnica ✅

La documentación es **correcta**:

- **VNNI no es un chip separado**: Son instrucciones AVX-512/AVX2 integradas en CPUs Intel
- **Generaciones soportadas**:
  - Ice Lake (10th gen, 2019) → AVX-512 VNNI
  - Tiger Lake (11th gen, 2020) → AVX-512 VNNI
  - Alder Lake+ (12th gen+, 2021+) → AVX2 VNNI
- **Aceleración real**: ~2-4x speedup en operaciones INT8 vs FP32
- **OpenVINO auto-detecta VNNI**: No requiere código especial, OpenVINO usa VNNI automáticamente si está disponible

### 2. Estrategia de Balanceo CPU/GPU ✅

El enfoque híbrido propuesto es **válido y recomendable**:

```
Pipeline Dual:
├── Segmentation → GPU (FP16) - modelo más pesado, se beneficia del paralelismo GPU
└── Pose → CPU (INT8/VNNI) - modelo más ligero, baja latencia en CPU
```

**Ventajas verificadas:**
- Reduce saturación de GPU
- Mejor utilización de recursos
- Menor latencia total del pipeline
- OpenVINO soporta diferentes devices por modelo nativamente

### 3. Estado Actual del Código

| Componente | Estado | Notas |
|------------|--------|-------|
| `Precision.INT8` enum | ✅ Definido | `bakery/core/entities/model_config.py:38-43` |
| `InferenceEngine` compilación | ✅ Soporta INT8 | Device-agnostic, funciona con cualquier precisión |
| `DualModelPipeline` | ✅ Compatible | Puede usar INT8 CPU + FP16 GPU sin cambios |
| Scripts de verificación | ✅ Completos | 7 scripts en `scripts/int8_vnni/` |
| Documentación | ✅ Detallada | `.docs/next/int8_vnni_cpu_load_balancing/` |
| **ModelRepository** | ⚠️ Gap | **No detecta INT8 automáticamente** |

### 4. Gap Identificado

**Archivo:** `bakery/adapters/openvino/model_repository.py:176-177`

```python
# Código actual:
precision = Precision.FP16 if "fp16" in str(model_path).lower() else Precision.FP32

# Debería ser:
path_str = str(model_path).lower()
if "int8" in path_str:
    precision = Precision.INT8
elif "fp16" in path_str:
    precision = Precision.FP16
else:
    precision = Precision.FP32
```

**Impacto:** Sin este cambio, modelos con nombres como `yolov11n-seg_int8.xml` son clasificados erróneamente como FP32.

---

## Plan de Implementación

### Fase 1: Corrección del Gap (Mínimo Necesario)

**Archivo a modificar:** `bakery/adapters/openvino/model_repository.py`

**Cambio:** Actualizar `_create_model_config()` (líneas 176-177) para detectar INT8:

```python
# Detect precision from model path (int8 vs fp16 vs fp32)
path_str = str(model_path).lower()
if "int8" in path_str:
    precision = Precision.INT8
elif "fp16" in path_str:
    precision = Precision.FP16
else:
    precision = Precision.FP32
```

### Fase 2: Actualizar Tests

**Archivo:** `bakery/tests/test_model_repository.py`

**Agregar:** Test case para detección de INT8:
- Crear modelo mock con "int8" en path
- Verificar que `precision == Precision.INT8`

### Fase 3: Integración con Wiki

**Actualizar:** `.docs/wiki/6.2-model-format-requirements.md`

**Agregar:** Sección sobre precisión INT8:
- Marcadores de precisión: `int8`, `fp16`, `fp32`
- Naming convention para modelos INT8

---

## Scripts Ya Disponibles

Los scripts de verificación/benchmark **ya existen** y están listos para uso:

| Script | Propósito | Comando |
|--------|-----------|---------|
| `verify_vnni.py` | Verificar soporte VNNI en CPU | `uv run scripts/int8_vnni/verify_vnni.py` |
| `verify_int8_execution.py` | Verificar ejecución INT8 real | `uv run scripts/int8_vnni/verify_int8_execution.py` |
| `calibrate_int8.py` | Calibrar modelos con NNCF | `uv run scripts/int8_vnni/calibrate_int8.py` |
| `benchmark_hybrid_cpu_gpu.py` | Benchmark pipeline híbrido | `uv run scripts/int8_vnni/benchmark_hybrid_cpu_gpu.py` |
| `verify_load_balancing.py` | Monitorear carga en tiempo real | `uv run scripts/int8_vnni/verify_load_balancing.py` |

---

## Recomendaciones Adicionales

### Consideraciones de Performance

1. **INT8 sin calibración puede ser más lento que FP16** - Los benchmarks existentes lo confirman
2. **Usar modelos calibrados con NNCF** para mejor accuracy y speedup
3. **Resolución óptima por device:**
   - CPU (INT8/VNNI): 320x320 (~78-85 FPS)
   - GPU (FP16): 640x640 (~22-25 FPS)

### Mejores Prácticas

1. Siempre ejecutar `verify_vnni.py` antes de usar INT8 en CPU
2. Para producción, calibrar modelos con `calibrate_int8.py`
3. Usar `benchmark_hybrid_cpu_gpu.py` para encontrar configuración óptima

---

## Verificación

Después de implementar el cambio:

1. **Test unitario:**
   ```bash
   uv run pytest bakery/tests/test_model_repository.py -v
   ```

2. **Verificar detección INT8:**
   ```bash
   # Crear modelo de prueba en path con "int8"
   # Verificar que ModelRepository detecta Precision.INT8
   ```

3. **Verificar pipeline híbrido:**
   ```bash
   uv run scripts/int8_vnni/verify_load_balancing.py --seg-device GPU --pose-device CPU
   ```

---

## Conclusión

El plan propuesto en `.docs/next/int8_vnni_cpu_load_balancing/` es:

- ✅ **Técnicamente correcto** - VNNI, INT8, y balanceo de carga están bien explicados
- ✅ **Viable** - La arquitectura actual ya soporta INT8
- ✅ **Bien documentado** - Scripts y guías de uso completos
- ⚠️ **Requiere un cambio menor** - Agregar detección de INT8 en ModelRepository

**Esfuerzo estimado:** Cambio mínimo (~5 líneas de código + 1 test case).
