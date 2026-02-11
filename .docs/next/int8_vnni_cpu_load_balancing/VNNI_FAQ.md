# FAQ: VNNI y Uso Automático

## ¿Hay que instalar algún paquete adicional?

**No, ya tienes todo instalado.**

Los paquetes necesarios ya están en `pyproject.toml`:
- ✅ `openvino==2024.0.0` - Runtime de OpenVINO
- ✅ `openvino-dev==2024.0.0` - Herramientas de desarrollo
- ✅ `nncf>=2.12` - Para calibración INT8 (opcional pero recomendado)
- ✅ `psutil>=7.1.2` - Para monitoreo de sistema

**VNNI no requiere paquetes adicionales** - es una característica del CPU, no un software.

### Verificar instalación

```bash
uv run scripts/int8_vnni/check_vnni_requirements.py
```

---

## ¿Cómo confirmar que realmente usa VNNI durante la inferencia?

### 1. Verificar que el CPU tiene VNNI

```bash
uv run scripts/int8_vnni/verify_vnni.py
```

**Salida esperada:**
```
✅ VNNI HABILITADO
   Tu CPU tiene aceleración hardware de INT8.
```

### 2. Verificar que el modelo es realmente INT8

```bash
uv run verify_int8_execution.py --device CPU --model yolov11n --resolution 320
```

**Salida esperada:**
```
✅ MODELO ES REALMENTE INT8
   La mayoría de operaciones usan INT8.
✅ CPU TIENE VNNI EN HARDWARE (aceleración automática de INT8)
   OpenVINO usará VNNI automáticamente durante la inferencia
```

### 3. Verificar durante la inferencia real

El script `verify_int8_execution.py` ahora verifica:
- ✅ Si el modelo tiene operaciones INT8 (>70% = realmente INT8)
- ✅ Si el CPU tiene VNNI en hardware (lee `/proc/cpuinfo`)
- ✅ Si OpenVINO reporta soporte INT8

**Si todas estas condiciones se cumplen, VNNI se está usando automáticamente.**

---

## ¿Tengo que especificar VNNI o es automático?

### ✅ **ES AUTOMÁTICO - NO HAY QUE ESPECIFICARLO**

OpenVINO detecta y usa VNNI automáticamente si:
1. Tu CPU tiene VNNI (verificado con `verify_vnni.py`)
2. Estás usando un modelo INT8
3. Estás ejecutando en device `CPU`

### Cómo funciona

```python
import openvino as ov

core = ov.Core()
model = core.read_model("model_int8.xml")

# Solo especificas CPU - OpenVINO usa VNNI automáticamente
compiled = core.compile_model(model, "CPU")

# Durante la inferencia, OpenVINO detecta que:
# - Modelo es INT8
# - CPU tiene VNNI
# → Usa VNNI automáticamente para acelerar
result = compiled([input_tensor])
```

**No hay flags ni configuraciones especiales necesarias.**

### Verificación automática

El script `verify_int8_execution.py` ahora:
- ✅ Detecta VNNI en hardware automáticamente
- ✅ Confirma que OpenVINO lo usará
- ✅ Muestra mensaje claro: "OpenVINO usará VNNI automáticamente para INT8"

---

## Resumen

| Pregunta | Respuesta |
|----------|-----------|
| ¿Paquetes adicionales? | ❌ No, ya están instalados |
| ¿Cómo confirmar uso? | ✅ `verify_int8_execution.py --device CPU` |
| ¿Es automático? | ✅ Sí, OpenVINO lo detecta y usa automáticamente |
| ¿Hay que especificarlo? | ❌ No, solo usa `device="CPU"` con modelo INT8 |

---

## Ejemplo Completo

```python
import openvino as ov
import numpy as np

# 1. Cargar modelo INT8
core = ov.Core()
model = core.read_model("exports/int8/yolov11n_320_int8.xml")

# 2. Compilar en CPU (VNNI se usa automáticamente si está disponible)
compiled = core.compile_model(model, "CPU")

# 3. Inferencia (VNNI acelera automáticamente)
input_tensor = np.random.rand(1, 3, 320, 320).astype(np.float32)
result = compiled([input_tensor])

# ✅ Si tu CPU tiene VNNI, OpenVINO lo usó automáticamente
# No necesitas hacer nada más
```

---

## Troubleshooting

**Problema**: No estoy seguro si VNNI se está usando

**Solución**:
1. Verificar CPU: `uv run scripts/int8_vnni/verify_vnni.py`
2. Verificar modelo: `uv run verify_int8_execution.py --device CPU`
3. Si ambos muestran ✅, VNNI se está usando automáticamente

**Problema**: OpenVINO no reporta "VNNI" explícitamente

**Solución**: Esto es normal. OpenVINO no siempre reporta "VNNI" en capabilities, pero si:
- Tu CPU tiene VNNI (verificado en `/proc/cpuinfo`)
- El modelo es INT8
- Estás usando `device="CPU"`

Entonces OpenVINO **SÍ está usando VNNI** automáticamente.

---

*Última actualización: 2026-02-09*
