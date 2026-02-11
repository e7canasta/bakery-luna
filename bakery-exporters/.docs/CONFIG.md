# Bakery Exporters Configuration

## Overview

Bakery Exporters supports two configuration modes:

1. **Embedded Mode**: Used within bakery-luna (uses `bakery.catalog.config`)
2. **Standalone Mode**: Independent package usage (uses `bakery_exporters.config`)

This allows maximum flexibility: reuse bakery-luna config when embedded, or have independent config when standalone.

## Configuration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Application                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Mode 1: Embedded in bakery-luna                       │
│  ├─ Imports: from bakery_exporters import ...         │
│  ├─ Uses: bakery.catalog.config                        │
│  └─ Benefit: Single source of truth                    │
│                                                          │
│  Mode 2: Standalone package                           │
│  ├─ Imports: from bakery_exporters import config      │
│  ├─ Uses: ExportersConfig                             │
│  └─ Benefit: No bakery-luna dependency                │
│                                                          │
└─────────────────────────────────────────────────────────┘
         ↓                             ↓
    ┌────────────────┐        ┌──────────────────┐
    │ bakery.catalog │        │ bakery_exporters │
    │    .config     │        │     .config      │
    ├────────────────┤        ├──────────────────┤
    │ • models_dir   │        │ • models_dir     │
    │ • calib_dir    │        │ • calib_dir      │
    │ • onnx_opset   │        │ • onnx_opset     │
    │ • default_yolo │        │ • default_yolo   │
    │                │        │ • cache_dir      │
    │                │        │ • temp_dir       │
    │                │        │ • int8_preset    │
    │                │        │ • log_level      │
    └────────────────┘        └──────────────────┘
         ↑                             ↑
    Shared via .env/.env file    Standalone via .env
```

## Mode 1: Embedded in bakery-luna

When using bakery-exporters within bakery-luna:

```python
from bakery_exporters import ExportPipeline, ExportFormat
from bakery.catalog import config  # Uses catalog config

pipeline = ExportPipeline()
results = pipeline.export(
    yolo_version="26",
    size="n",
    task="segmentation",
    resolution=320,
    formats=[ExportFormat.FP16]
)
```

**Configuration Source:**
- Uses `bakery.catalog.BakeryConfig`
- Env vars: `BAKERY_MODELS_DIR`, `BAKERY_CALIBRATION_DIR`, `BAKERY_ONNX_OPSET`, `BAKERY_DEFAULT_YOLO`
- File: `.env` in project root

**Advantages:**
- Single configuration source
- Shared across entire bakery-luna
- Easy deployment with consistent settings

## Mode 2: Standalone Usage

When using bakery-exporters independently:

```python
from bakery_exporters import ExportPipeline, ExportFormat, config

# config is ExportersConfig instance
print(f"Models dir: {config.models_dir}")

# Override at runtime if needed
config.models_dir = Path("/custom/models")
config.ensure_dirs()

pipeline = ExportPipeline()
results = pipeline.export(...)
```

**Configuration Sources (in order of priority):**
1. Runtime assignment: `config.models_dir = Path(...)`
2. Environment variables: `BAKERY_EXPORTERS_*`
3. .env file in current directory (searches up 3 levels)
4. Default values

**Environment Variables:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `BAKERY_MODELS_DIR` | `./models` | Exported models directory |
| `BAKERY_CALIBRATION_DIR` | `./calibration_data` | Calibration frames directory |
| `BAKERY_ONNX_OPSET` | `17` | ONNX opset version |
| `BAKERY_DEFAULT_YOLO` | `11` | Default YOLO version |
| `BAKERY_EXPORTERS_CACHE_DIR` | `./.cache/exporters` | ONNX cache during conversion |
| `BAKERY_EXPORTERS_TEMP_DIR` | `./.tmp/bakery` | Temporary files |
| `BAKERY_INT8_SYNTHETIC_SAMPLES` | `100` | Synthetic calibration samples |
| `BAKERY_INT8_PRESET` | `MIXED` | INT8 quantization preset |
| `BAKERY_EXPORTERS_LOG_LEVEL` | `INFO` | Logging level |
| `BAKERY_EXPORTERS_VERBOSE` | `false` | Verbose output |

**Example .env file (standalone):**

```bash
# Shared with catalog
BAKERY_MODELS_DIR=/data/ml/models
BAKERY_CALIBRATION_DIR=/data/ml/calibration
BAKERY_ONNX_OPSET=17

# Exporters-specific
BAKERY_EXPORTERS_CACHE_DIR=/tmp/bakery-cache
BAKERY_EXPORTERS_TEMP_DIR=/tmp/bakery
BAKERY_INT8_PRESET=MIXED
BAKERY_EXPORTERS_LOG_LEVEL=DEBUG
```

**Advantages:**
- No dependencies on bakery-luna
- Can be used in other projects
- Fine-grained control over exporter settings

## Configuration via API

### Standalone Configuration

```python
from bakery_exporters import config
from pathlib import Path

# Check current configuration
print(config.to_dict())

# Override specific settings
config.models_dir = Path("/custom/models")
config.onnx_opset = 18
config.int8_preset = "PERFORMANCE"
config.verbose = True

# Ensure all directories exist
config.ensure_dirs()
```

### Creating Custom Config

```python
from bakery_exporters import ExportersConfig
from pathlib import Path

# Create custom config instance
custom_config = ExportersConfig(
    models_dir=Path("/models"),
    calibration_dir=Path("/calibration"),
    cache_dir=Path("/cache"),
    onnx_opset=18,
    int8_preset="PERFORMANCE"
)

custom_config.ensure_dirs()
```

### Using Catalog Config

```python
from bakery.catalog import config as catalog_config
from bakery_exporters import ExportersConfig

# Create ExportersConfig from catalog config
exporter_config = ExportersConfig.from_catalog(catalog_config)
```

## Configuration Priority

When bakery-exporters looks for configuration:

1. **Runtime assignment** (highest priority)
   ```python
   config.models_dir = Path("/custom")
   ```

2. **Environment variables**
   ```bash
   export BAKERY_MODELS_DIR=/custom
   ```

3. **.env file** (searches up 3 parent directories)
   ```bash
   BAKERY_MODELS_DIR=/custom
   ```

4. **Default values** (lowest priority)
   ```python
   models_dir: Path = Path("./models")
   ```

## Integration with bakery.catalog

**When embedded:**

```python
# Option 1: Use catalog config directly
from bakery.catalog import config
from bakery_exporters import ExportPipeline

pipeline = ExportPipeline()
# Automatically uses config.models_dir, config.onnx_opset, etc.

# Option 2: Convert catalog config
from bakery.catalog import config as catalog_config
from bakery_exporters import ExportersConfig

exporter_config = ExportersConfig.from_catalog(catalog_config)
```

**When standalone:**

```python
# ExportersConfig is independent
from bakery_exporters import config

# No bakery-luna dependency needed
config.models_dir = Path("./models")
```

## Common Configuration Scenarios

### Scenario 1: Development (Standalone)

```bash
# .env
BAKERY_MODELS_DIR=./models
BAKERY_CALIBRATION_DIR=./calibration_data
BAKERY_EXPORTERS_VERBOSE=true
BAKERY_EXPORTERS_LOG_LEVEL=DEBUG
```

### Scenario 2: Production (Standalone)

```bash
# .env
BAKERY_MODELS_DIR=/data/models
BAKERY_CALIBRATION_DIR=/data/calibration
BAKERY_EXPORTERS_CACHE_DIR=/var/cache/bakery
BAKERY_EXPORTERS_TEMP_DIR=/var/tmp/bakery
BAKERY_EXPORTERS_LOG_LEVEL=WARNING
```

### Scenario 3: Embedded in bakery-luna

```bash
# Single .env for entire bakery-luna
BAKERY_MODELS_DIR=/data/ml/models
BAKERY_CALIBRATION_DIR=/data/ml/calibration
BAKERY_ONNX_OPSET=17
BAKERY_DEFAULT_YOLO=11

# bakery-exporters inherits all these
```

### Scenario 4: Docker

```dockerfile
FROM python:3.11

WORKDIR /app

# Install dependencies
RUN pip install bakery-exporters[full]

# Configuration via environment
ENV BAKERY_MODELS_DIR=/models
ENV BAKERY_CALIBRATION_DIR=/calibration
ENV BAKERY_EXPORTERS_CACHE_DIR=/cache

# Create directories
RUN mkdir -p /models /calibration /cache

CMD ["python", "export_script.py"]
```

## Best Practices

### ✅ Do's

1. **Use catalog config when embedded**
   - Single source of truth
   - Consistent across bakery-luna

2. **Set environment variables in CI/CD**
   - Flexible deployment
   - No hardcoded paths

3. **Call `config.ensure_dirs()`** at startup
   - Prevents runtime errors
   - Explicit dependency on directories

4. **Use `config.to_dict()`** for debugging
   - Verify current configuration
   - Log configuration state

### ❌ Don'ts

1. **Don't hardcode paths** in code
   - Use configuration instead
   - Makes code portable

2. **Don't assume directories exist**
   - Call `config.ensure_dirs()`
   - Handle missing directories gracefully

3. **Don't mix config sources**
   - Pick one: env vars OR runtime assignment
   - Avoid confusion

## Troubleshooting

### Issue: Wrong models directory

```python
from bakery_exporters import config
print(f"Current: {config.models_dir}")  # Check current value
config.ensure_dirs()  # Create if missing
```

### Issue: Environment variables not picked up

1. Check if variables are exported:
   ```bash
   echo $BAKERY_MODELS_DIR
   ```

2. Check if .env file exists and is readable
   ```bash
   ls -la .env
   ```

3. Set explicitly at runtime:
   ```python
   from bakery_exporters import config
   from pathlib import Path
   config.models_dir = Path("/path")
   ```

### Issue: Permissions error on cache/temp directories

```python
from bakery_exporters import config

# Specify writable directories
config.cache_dir = Path("/tmp/bakery-cache")
config.temp_dir = Path("/tmp/bakery-temp")
config.ensure_dirs()
```

## Related

- [Specification](specs/exporters-specification.md) - API reference
- [USAGE.md](../USAGE.md) - Usage guide
- [Examples](../examples/) - Code examples
- [bakery.catalog.config](../../bakery/catalog/config.py) - Catalog configuration
