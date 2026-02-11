# Model Catalog Specification

**Version:** 1.0.0
**Last Updated:** 2025-02-10
**Related ADR:** [ADR-005](/docs/adrs/005-model-catalog-configuration.md)

## Overview

El Model Catalog es el sistema centralizado de gestión de modelos exportados en Bakery. Proporciona:

- Estructura de directorios normalizada
- Configuración via variables de entorno
- API Python para descubrimiento y paths
- CLI consistente en todos los scripts de export

## Directory Structure

```
$BAKERY_MODELS_DIR/                    # default: models/
├── yolo26n-seg/                       # {model_name}
│   ├── 320/                           # {resolution}
│   │   ├── onnx/
│   │   │   └── model.onnx             # ONNX format (FP32)
│   │   ├── fp16/
│   │   │   ├── model.xml              # OpenVINO IR
│   │   │   └── model.bin              # OpenVINO weights
│   │   ├── int8/
│   │   │   ├── model.xml              # INT8 (synthetic calibration)
│   │   │   └── model.bin
│   │   └── int8_calibrated/
│   │       ├── model.xml              # INT8 (real data calibration)
│   │       └── model.bin
│   └── 640/
│       └── ...
├── yolo26m-pose/
│   └── 256/
│       └── ...
└── yolo11n/
    └── 320/
        └── ...
```

### Naming Conventions

| Component | Format | Examples |
|-----------|--------|----------|
| Model Name | `yolo{version}{size}[-task]` | `yolo26n`, `yolo11m-seg`, `yolo26l-pose` |
| Resolution | `{pixels}` | `320`, `640`, `256` |
| Format | `onnx`, `fp16`, `int8`, `int8_calibrated` | - |
| Files | `model.{ext}` | `model.onnx`, `model.xml`, `model.bin` |

### Task Suffixes

| Task | Suffix | Example |
|------|--------|---------|
| Detection | (none) | `yolo26n` |
| Segmentation | `-seg` | `yolo26n-seg` |
| Pose | `-pose` | `yolo26n-pose` |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BAKERY_MODELS_DIR` | `models` | Base directory for exported models |
| `BAKERY_CALIBRATION_DIR` | `calibration_data` | Directory for calibration frames |
| `BAKERY_DEFAULT_YOLO` | `11` | Default YOLO version |
| `BAKERY_ONNX_OPSET` | `17` | ONNX opset for OpenVINO compatibility |

### .env File

```bash
# .env (copy from .env.example)
BAKERY_MODELS_DIR=models
BAKERY_CALIBRATION_DIR=calibration_data
BAKERY_DEFAULT_YOLO=11
BAKERY_ONNX_OPSET=17
```

## Python API

### Import

```python
from bakery.config import config, ModelPath
from bakery.config import YOLO_VERSIONS, MODEL_SIZES, RESOLUTIONS, TASK_TYPES
```

### BakeryConfig

```python
# Global config instance
config.models_dir          # Path("models")
config.calibration_dir     # Path("calibration_data")
config.default_yolo_version # "11"
config.onnx_opset          # 17

# Override at runtime
config.models_dir = Path("/custom/path")
```

### ModelPath

#### get_model_name()

Build model name from components.

```python
ModelPath.get_model_name("26", "n", "segmentation")
# Returns: "yolo26n-seg"

ModelPath.get_model_name("11", "m", "pose")
# Returns: "yolo11m-pose"

ModelPath.get_model_name("8", "l", "detection")
# Returns: "yolo8l"
```

#### parse_model_name()

Parse model name into components.

```python
ModelPath.parse_model_name("yolo26n-seg")
# Returns: {"yolo_version": "26", "size": "n", "task": "segmentation"}

ModelPath.parse_model_name("yolo11m-pose")
# Returns: {"yolo_version": "11", "size": "m", "task": "pose"}
```

#### get()

Get path to model file.

```python
ModelPath.get("yolo26n-seg", 320, "fp16")
# Returns: Path("models/yolo26n-seg/320/fp16/model.xml")

ModelPath.get("yolo26n-seg", 320, "onnx")
# Returns: Path("models/yolo26n-seg/320/onnx/model.onnx")
```

#### build()

Build path and create directories.

```python
path = ModelPath.build("yolo26n-seg", 320, "int8")
# Creates: models/yolo26n-seg/320/int8/
# Returns: Path("models/yolo26n-seg/320/int8/model.xml")
```

#### exists()

Check if model exists.

```python
ModelPath.exists("yolo26n-seg", 320, "fp16")
# Returns: True/False (checks both .xml and .bin for OpenVINO)
```

#### list_models()

List available models.

```python
models = ModelPath.list_models(task="segmentation")
# Returns: [
#   {"name": "yolo26n-seg", "resolution": 320, "formats": ["onnx", "fp16"], ...},
#   {"name": "yolo26m-seg", "resolution": 320, "formats": ["fp16", "int8"], ...},
# ]
```

## CLI Reference

### Export Scripts

All export scripts share common flags:

| Flag | Short | Description |
|------|-------|-------------|
| `--model` | `-m` | YOLO version (8, 11, 26) |
| `--model-size` | `-s` | Model size (n, s, m, l, x) |
| `--resolution` | `-r` | Input resolution |
| `--format` | `-f` | Output format(s) |
| `--output` | `-o` | Override models directory |
| `--dry-run` | - | Show plan without executing |

### export_sauron_segmentation.py

Export segmentation models to FP16 (GPU).

```bash
# Basic usage
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n --resolution 320

# Multiple sizes and resolutions
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n s m --resolution 256 320

# Include ONNX
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n --resolution 320 --format onnx fp16
```

### export_sauron_pose.py

Export pose models to FP16 (GPU).

```bash
uv run scripts/tasks/export/export_sauron_pose.py \
  --model 26 --model-size m --resolution 256
```

### export_int8.py

Export models to INT8 (CPU/VNNI).

```bash
# Segmentation INT8
uv run scripts/tasks/export/export_int8.py \
  --model 26 --model-size n --type segmentation --resolution 320

# Pose INT8
uv run scripts/tasks/export/export_int8.py \
  --model 26 --model-size m --type pose --resolution 256

# Detection INT8
uv run scripts/tasks/export/export_int8.py \
  --model 11 --model-size n --type detection --resolution 320
```

### calibrate_int8.py

Calibrate INT8 models with real data.

```bash
# Calibrate specific model
uv run scripts/tasks/export/calibrate_int8.py \
  --model yolo26n-seg --resolution 320

# Calibrate all ONNX models
uv run scripts/tasks/export/calibrate_int8.py --all

# Calibrate all at specific resolution
uv run scripts/tasks/export/calibrate_int8.py --all --resolution 320
```

## Usage Examples

### Export Workflow

```bash
# 1. Configure (optional)
cp .env.example .env
# Edit BAKERY_MODELS_DIR if needed

# 2. Export FP16 for GPU
uv run scripts/tasks/export/export_sauron_segmentation.py \
  -m 26 -s n s m -r 320

# 3. Export INT8 for CPU
uv run scripts/tasks/export/export_int8.py \
  -m 26 -s n s m -t segmentation -r 320

# 4. Calibrate with real data (optional, better accuracy)
uv run scripts/tasks/export/calibrate_int8.py --all -r 320
```

### Load Model in Code

```python
from bakery.config import ModelPath
import openvino as ov

# Get model path
model_path = ModelPath.get("yolo26n-seg", 320, "fp16")

# Load with OpenVINO
core = ov.Core()
model = core.read_model(model_path)
compiled = core.compile_model(model, "GPU")
```

### List Available Models

```python
from bakery.config import ModelPath

# All segmentation models
for m in ModelPath.list_models(task="segmentation"):
    print(f"{m['name']} @ {m['resolution']}px: {m['formats']}")

# Check if specific model exists
if ModelPath.exists("yolo26n-seg", 320, "int8_calibrated"):
    print("Calibrated model available!")
```

## Testing

```bash
# Test config module
uv run python -c "
from bakery.config import config, ModelPath
print(f'Models dir: {config.models_dir}')
print(f'Model path: {ModelPath.get(\"yolo26n-seg\", 320, \"fp16\")}')
"

# Dry-run export
uv run scripts/tasks/export/export_sauron_segmentation.py \
  -m 26 -s n -r 320 --dry-run
```

## Related

- **ADR:** [ADR-005: Model Catalog Configuration](/docs/adrs/005-model-catalog-configuration.md)
- **Code:** [bakery/config.py](/bakery/config.py)
- **Template:** [.env.example](/.env.example)
- **Export Scripts:** [scripts/tasks/export/](/scripts/tasks/export/)
