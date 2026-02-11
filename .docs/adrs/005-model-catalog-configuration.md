# ADR-005: Model Catalog Configuration System

**Status:** Accepted
**Date:** 2025-02-10
**Authors:** Claude, User

## Context

El proyecto Bakery tiene múltiples scripts de exportación de modelos (`export_sauron_segmentation.py`, `export_sauron_pose.py`, `export_int8.py`, `calibrate_int8.py`) que:

1. **Duplicaban configuración**: Cada script definía sus propias constantes para versiones YOLO, tamaños, resoluciones
2. **Paths hardcodeados**: Directorios de salida (`exports/`, `models/`) estaban dispersos y sin consistencia
3. **Nomenclatura inconsistente**: Mezcla de `medium`, `wide`, `320px`, `fp32/`, `int8/` sin patrón claro
4. **Sin configuración externa**: No había forma de cambiar paths sin modificar código

Esto hacía difícil:
- Mantener sincronizados los scripts
- Desplegar en diferentes entornos (dev, prod, CI)
- Entender la estructura del repositorio de modelos

## Decision

Implementar un **sistema de configuración centralizado** inspirado en gestores de paquetes (npm, Maven, Roboflow):

### 1. Módulo de Configuración (`bakery/config.py`)

```python
from bakery.config import config, ModelPath

# Configuración global cargada desde env vars / .env
config.models_dir          # Path: models/
config.calibration_dir     # Path: calibration_data/
config.default_yolo_version # str: "11"
config.onnx_opset          # int: 17
```

### 2. Estructura de Directorios Normalizada

```
$BAKERY_MODELS_DIR/
└── {model_name}/          # ej: yolo26n-seg
    └── {resolution}/      # ej: 320
        ├── onnx/
        │   └── model.onnx
        ├── fp16/
        │   ├── model.xml
        │   └── model.bin
        ├── int8/
        │   ├── model.xml
        │   └── model.bin
        └── int8_calibrated/
            ├── model.xml
            └── model.bin
```

### 3. Variables de Entorno

```bash
BAKERY_MODELS_DIR=models
BAKERY_CALIBRATION_DIR=calibration_data
BAKERY_DEFAULT_YOLO=11
BAKERY_ONNX_OPSET=17
```

### 4. API de ModelPath

```python
# Construir nombre de modelo
ModelPath.get_model_name("26", "n", "segmentation")  # "yolo26n-seg"

# Obtener path a modelo
ModelPath.get("yolo26n-seg", 320, "fp16")  # models/yolo26n-seg/320/fp16/model.xml

# Crear directorios y obtener path
ModelPath.build("yolo26n-seg", 320, "int8")  # crea dirs, retorna path

# Verificar existencia
ModelPath.exists("yolo26n-seg", 320, "fp16")  # bool

# Listar modelos disponibles
ModelPath.list_models(task="segmentation")  # [{name, resolution, formats, ...}]
```

## Consequences

### Positive

- **Single Source of Truth**: Una sola definición de YOLO_VERSIONS, MODEL_SIZES, RESOLUTIONS
- **Configuración por entorno**: `.env` para dev, env vars para prod/CI
- **Estructura predecible**: Fácil encontrar cualquier modelo por `{name}/{resolution}/{format}`
- **API consistente**: Todos los scripts usan `ModelPath` para paths
- **Retrocompatible**: `--output` flag permite override puntual

### Negative

- **Migración requerida**: Modelos existentes en estructura vieja necesitan reorganizarse
- **Dependencia adicional**: Scripts ahora dependen de `bakery.config`
- **Curva de aprendizaje**: Usuarios deben entender la nueva estructura

### Neutral

- Los scripts mantienen la misma interfaz CLI
- La estructura de directorios es más profunda pero más organizada

## Alternatives Considered

### Alternative 1: YAML/JSON Config File

Usar un archivo `bakery.yaml` o `config.json` para toda la configuración.

**Rechazado porque:**
- Añade dependencia de parsing (pyyaml)
- Menos flexible que env vars para CI/CD
- `.env` es patrón más común en Python

### Alternative 2: Mantener Paths en CLI

Seguir pasando todos los paths como argumentos CLI.

**Rechazado porque:**
- Comandos muy largos
- Fácil equivocarse de path
- No hay defaults sensatos

### Alternative 3: XDG Base Directory Spec

Usar `~/.local/share/bakery/models` como npm usa `node_modules`.

**Rechazado porque:**
- Proyecto es local, no sistema
- Queremos modelos junto al código
- Más difícil de versionar/backupear

## References

- [bakery/config.py](/bakery/config.py) - Implementación
- [.env.example](/.env.example) - Template de configuración
- [npm package.json](https://docs.npmjs.com/cli/v9/configuring-npm/package-json) - Inspiración
- [Maven Repository Layout](https://maven.apache.org/guides/introduction/introduction-to-repositories.html) - Inspiración
