
# Int8 inference and VNNI chip


 VNNI (Vector Neural Network Instructions) es aceleración hardware de INT8 en CPU Intel (no un chip separado, sino instrucciones del CPU). Usar CPU con VNNI para INT8 puede reducir la carga en GPU y CPU, especialmente en un pipeline dual (segmentation + pose).


# Plan: INT8/VNNI y Balanceo de Carga CPU/GPU

## Objetivo

Recopilar y organizar los scripts existentes de verificación INT8/VNNI, generar pruebas adicionales para validar balanceo de carga entre CPU (INT8/VNNI) y GPU, y crear documentación markdown que explique cómo usar CPU con VNNI para reducir carga en GPU y CPU en pipelines duales.

## Contexto Técnico

**VNNI (Vector Neural Network Instructions)**:

- No es un chip separado, sino instrucciones hardware del CPU Intel
- Disponible desde Ice Lake (10th gen, 2019+) con AVX-512 VNNI
- Alder Lake+ (12th gen, 2021+) con AVX2 VNNI
- Acelera operaciones INT8 en CPU (~2-4x speedup vs FP32)

**Estrategia de Balanceo**:

- Pipeline dual (segmentation + pose) puede distribuir carga:
- **CPU (INT8/VNNI)**: Un modelo (ej: pose estimation)
- **GPU (FP16)**: Otro modelo (ej: segmentation)
- Reduce saturación de GPU y libera CPU para otras tareas
- OpenVINO soporta esto nativamente con diferentes devices por modelo

## Archivos Existentes a Recopilar

### Scripts de Verificación

- `verify_vnni.py` - Verifica soporte VNNI en CPU
- `verify_int8_execution.py` - Verifica ejecución real de INT8
- `calibrate_int8.py` - Calibración INT8 con NNCF
- `benchmark_cross_device.py` - Benchmarks CPU vs GPU
- `benchmark_cpu.py` - Benchmarks específicos de CPU

### Scripts de Inferencia

- `inference_int8.py` - Inferencia INT8 (CPU/GPU)
- `inference_fp16.py` - Inferencia FP16 (GPU)
- `run_luna.py` - Pipeline dual (segmentation + pose)


### 1. Scripts en Estructura Clara

directorio `scripts/int8_vnni/` con:

- `verify_vnni.py` → `scripts/int8_vnni/verify_vnni.py`
- `verify_int8_execution.py` → `scripts/int8_vnni/verify_int8_execution.py`
- `calibrate_int8.py` → `scripts/int8_vnni/calibrate_int8.py`
- `benchmark_cross_device.py` → `scripts/int8_vnni/benchmark_cross_device.py`
- `benchmark_cpu.py` → `scripts/int8_vnni/benchmark_cpu.py` (si existe)



### 2. Script de Balanceo de Carga

`scripts/int8_vnni/benchmark_hybrid_cpu_gpu.py`:

- Ejecuta pipeline dual con:
- Modelo 1 en CPU (INT8/VNNI)
- Modelo 2 en GPU (FP16)
- Compara vs:
- Ambos en CPU
- Ambos en GPU
- Métricas:
- FPS total
- CPU% (por core)
- GPU utilization
- Memoria (RAM + VRAM)
- Temperatura

### 3. Script de Verificación de Carga

`scripts/int8_vnni/verify_load_balancing.py`:

- Monitorea carga en tiempo real durante inferencia híbrida
- Muestra:
- CPU cores utilizados
- GPU utilization
- Memory bandwidth
- Throughput por dispositivo
- Valida que la carga esté distribuida correctamente

### 4. Documentación

`docs/INT8_VNNI_LOAD_BALANCING.md` con:

#### Secciones:

1. **Introducción a VNNI**

- Qué es VNNI (no es chip separado)
- Generaciones Intel con soporte
- Cómo verificar soporte

2. **INT8 en CPU vs GPU**

- Ventajas de INT8 en CPU (VNNI)
- Cuándo usar CPU vs GPU
- Performance esperada

3. **Estrategias de Balanceo de Carga**

- Pipeline dual: CPU (INT8) + GPU (FP16)
- Casos de uso
- Configuración en OpenVINO

4. **Guía de Uso**

- Verificar VNNI: `uv run verify_vnni.py`
- Verificar INT8 real: `uv run verify_int8_execution.py`
- Benchmark híbrido: `uv run benchmark_hybrid_cpu_gpu.py`
- Monitorear carga: `uv run verify_load_balancing.py`

5. **Resultados y Recomendaciones**

- Benchmarks típicos
- Mejores prácticas
- Troubleshooting


## Validación

1. Ejecutar `verify_vnni.py` - debe detectar VNNI si está disponible
2. Ejecutar `verify_int8_execution.py` - debe confirmar ejecución INT8 real
3. Ejecutar `benchmark_hybrid_cpu_gpu.py` - debe mostrar distribución de carga
4. Revisar documentación generada - debe ser clara y completa

## Notas Técnicas

- VNNI requiere CPU Intel Ice Lake+ (2019+)
- OpenVINO detecta VNNI automáticamente si está disponible
- INT8 sin calibración puede ser más lento que FP16 (ver `BENCHMARK_FINAL.md`)
- Para mejor performance INT8, usar modelos calibrados con NNCF
- Pipeline dual puede usar diferentes devices por modelo en OpenVINO



--- 

### Pensando un poco en nuestro disenio actual el Impacto:


el códigobase permite configurar modelos con precisión INT8, lo que en CPUs Intel i7 con VNNI puede habilitar automáticamente los operadores de 8 bits, pero no hay delegación explícita por modelo en el código actual. [1](#1-0) 

## Detalles

### Soporte para INT8
- `Precision.INT8` está definido en `ModelConfig` y puede usarse al crear configuraciones de modelo. [1](#1-0) 
- Los tests muestran ejemplos con `precision="int8"` en `ModelConfig.from_args`. [2](#1-1) 

### Delegación y VNNI
- `InferenceEngine` compila modelos para CPU/GPU según `ModelConfig.device`, pero no hay lógica específica para forzar VNNI o delegar solo ciertos modelos a INT8. [3](#1-2) 
- OpenVINO podría habilitar VNNI automáticamente al compilar un modelo INT8 en una CPU compatible, pero el códigobase no lo gestiona explícitamente.

### Cómo habilitar INT8 para un modelo
1) Asegúrate de tener un modelo cuantizado a INT8 (exportado con OpenVINO o NNCF).
2) Configura `ModelConfig.precision = Precision.INT8` al descubrir o crear el modelo. [4](#1-3) 
3) El `InferenceEngine` usará esa configuración al compilar para CPU; OpenVINO activará VNNI si el hardware lo soporta. [5](#1-4) 

## Notes
- No hay una ruta en `ModelRepository` que detecte automáticamente INT8 desde el nombre de archivo (a diferencia de FP16). [6](#1-5) 
- Para delegar solo ciertos modelos a INT8, deberías extender `ModelRepository` o pasar configuraciones específicas al crear `ModelConfig`. [7](#1-6) 

- Adapters Layer 

### Citations

**File:** bakery/core/entities/model_config.py (L38-44)
```python
class Precision(Enum):
    """Model precision options."""

    FP16 = "fp16"
    FP32 = "fp32"
    INT8 = "int8"

```

**File:** bakery/core/entities/model_config.py (L94-133)
```python
    @classmethod
    def from_args(
        cls,
        model_path: Path,
        model_type: str,
        resolution: int,
        device: str,
        precision: str,
        confidence: float,
        model_size: Optional[str] = None,
    ) -> "ModelConfig":
        """
        Create ModelConfig from string arguments.

        Args:
            model_path: Path to model file
            model_type: Model type ("segmentation", "pose")
            resolution: Input resolution
            device: Device ("GPU", "CPU", "AUTO")
            precision: Precision ("fp16", "fp32", "int8")
            confidence: Confidence threshold
            model_size: Optional model size ("s", "m", "l", "x")

        Returns:
            ModelConfig instance

        Example:
            >>> config = ModelConfig.from_args(
            ...     Path("model.xml"), "segmentation", 256, "GPU", "fp16", 0.25, "l"
            ... )
        """
        return cls(
            model_path=model_path,
            model_type=ModelType(model_type),
            resolution=resolution,
            device=Device(device),
            precision=Precision(precision),
            confidence=confidence,
            model_size=ModelSize(model_size) if model_size else None,
        )
```

**File:** bakery/tests/test_entities.py (L311-312)
```python
        seg_config = ModelConfig.from_args(seg_path, "segmentation", 192, "CPU", "int8", 0.25, "s")
        pose_config = ModelConfig.from_args(pose_path, "pose", 256, "GPU", "fp16", 0.25, "l")
```

**File:** bakery/adapters/openvino/inference_engine.py (L42-72)
```python
    def _compile(self):
        """
        Compile model with device fallback (GPU → CPU).

        This method attempts to compile the model for the requested device.
        If GPU is requested but unavailable, it automatically falls back to CPU.

        Raises:
            RuntimeError: If compilation fails on all attempted devices
        """
        # Read model from file
        model = self.core.read_model(self.config.model_path)

        # Get available devices
        available_devices = self.core.available_devices

        # Attempt compilation with fallback logic
        target_device = self.config.device.value

        if target_device == "GPU" and "GPU" in available_devices:
            try:
                self.compiled_model = self.core.compile_model(model, "GPU")
                self.device = "GPU"
                return
            except RuntimeError:
                # GPU compilation failed, try CPU
                pass

        # Fallback to CPU (or if CPU was requested)
        self.compiled_model = self.core.compile_model(model, "CPU")
        self.device = "CPU"
```


## Ejemplo OpenVINO con INT8 (VNNI) en CPU

```python
from pathlib import Path
from bakery.core.entities.model_config import ModelConfig, Device, Precision, ModelType
from bakery.adapters.openvino.inference_engine import InferenceEngine

# 1) Configurar modelo INT8 para CPU
seg_config = ModelConfig.from_args(
    model_path=Path("models/seg_int8.xml"),
    model_type="segmentation",
    resolution=640,
    device="CPU",
    precision="int8",
    confidence=0.25,
    model_size="s"
)

# 2) Crear motor de inferencia (OpenVINO compilará para CPU)
seg_engine = InferenceEngine(seg_config)
print(f"Modelo listo en dispositivo: {seg_engine.get_device()}")
```

### Detalles

- `Precision.INT8` está definido en `ModelConfig` y se puede pasar como `"int8"` en `from_args`. [1](#2-0) 
- El test existente ya muestra un ejemplo con `precision="int8"` y `device="CPU"`. [2](#2-1) 
- `InferenceEngine._compile()` usará `device="CPU"` y OpenVINO habilitará VNNI automáticamente si el hardware lo soporta. [3](#2-2) 

### Opcional: detección automática de INT8 desde el path

Si quieres que `ModelRepository` detecte INT8 por nombre (como hace con FP16), podrías extender la lógica de detección de precisión:

```python
# En model_repository.py,类似 a FP16 detection
precision = (
    Precision.INT8 if "int8" in str(model_path).lower()
    else Precision.FP16 if "fp16" in str(model_path).lower()
    else Precision.FP32
)
```

Actualmente solo se detecta FP16 automáticamente. [4](#2-3) 

## Notes
- Asegúrate de tener un modelo cuantizado a INT8 (exportado con OpenVINO/NNCF). [5](#2-4) 
- No hay delegación explícita por modelo; se configura por `ModelConfig`. [6](#2-5) 

Wiki pages you might want to explore:
- [Overview (e7canasta/bakery-luna)](/wiki/e7canasta/bakery-luna#1)
- [Dual Model Pipeline (e7canasta/bakery-luna)](/wiki/e7canasta/bakery-luna#3.3)

### Citations

**File:** bakery/core/entities/model_config.py (L38-44)
```python
class Precision(Enum):
    """Model precision options."""

    FP16 = "fp16"
    FP32 = "fp32"
    INT8 = "int8"

```

**File:** bakery/core/entities/model_config.py (L94-133)
```python
    @classmethod
    def from_args(
        cls,
        model_path: Path,
        model_type: str,
        resolution: int,
        device: str,
        precision: str,
        confidence: float,
        model_size: Optional[str] = None,
    ) -> "ModelConfig":
        """
        Create ModelConfig from string arguments.

        Args:
            model_path: Path to model file
            model_type: Model type ("segmentation", "pose")
            resolution: Input resolution
            device: Device ("GPU", "CPU", "AUTO")
            precision: Precision ("fp16", "fp32", "int8")
            confidence: Confidence threshold
            model_size: Optional model size ("s", "m", "l", "x")

        Returns:
            ModelConfig instance

        Example:
            >>> config = ModelConfig.from_args(
            ...     Path("model.xml"), "segmentation", 256, "GPU", "fp16", 0.25, "l"
            ... )
        """
        return cls(
            model_path=model_path,
            model_type=ModelType(model_type),
            resolution=resolution,
            device=Device(device),
            precision=Precision(precision),
            confidence=confidence,
            model_size=ModelSize(model_size) if model_size else None,
        )
```
