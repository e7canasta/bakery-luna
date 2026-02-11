# Bakery Exporters

Export and conversion pipelines for Bakery YOLO models to optimized inference formats.

## Quick Start

```bash
# Install with all dependencies
pip install bakery-luna bakery-exporters[full]

# Export model to FP16 (GPU)
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n --resolution 320

# Or export to INT8 (CPU)
uv run scripts/tasks/export/export_int8.py \
  --model 26 --model-size n --type segmentation --resolution 320
```

## Installation

```bash
# Minimal (catalog only)
pip install bakery-luna

# With ONNX export support
pip install bakery-luna bakery-exporters[onnx]

# With OpenVINO conversion
pip install bakery-luna bakery-exporters[openvino]

# With INT8 quantization
pip install bakery-luna bakery-exporters[nncf]

# All dependencies (~800MB)
pip install bakery-luna bakery-exporters[full]
```

## Python API

```python
from bakery_exporters import ExportPipeline, ExportFormat

# Create pipeline
pipeline = ExportPipeline()

# Export to multiple formats
results = pipeline.export(
    yolo_version="26",
    size="n",
    task="segmentation",
    resolution=320,
    formats=[ExportFormat.ONNX, ExportFormat.FP16, ExportFormat.INT8]
)

# Check results
for result in results:
    print(f"{result.model_name}: {result.output_path}")
```

## CLI Usage

```bash
# Export segmentation model to FP16
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n --resolution 320

# Export to INT8
uv run scripts/tasks/export/export_int8.py \
  --model 26 --model-size n --type segmentation --resolution 320 --format int8

# Calibrate with real data for better accuracy
uv run scripts/tasks/export/calibrate_int8.py \
  --model yolo26n-seg --resolution 320

# Dry run (no files created)
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n --resolution 320 --dry-run
```

## Supported Export Formats

| Format | Use Case | Size | Speed-up | Export Time |
|--------|----------|------|----------|-------------|
| **ONNX** | Framework-agnostic | 100% | 1x | 40-60s |
| **FP16** | GPU inference | 50% | 2-3x | 5-10s |
| **INT8** (synthetic) | CPU quick | 25% | 4-6x | 20-30s |
| **INT8** (calibrated) | CPU production | 25% | 4-6x | 60-120s |

## Architecture

```
ExportPipeline (Orchestrator)
├── OnnxExporter
│   └── Export YOLO → ONNX (via ultralytics)
├── OpenVINOConverter
│   ├── to_fp16() → Compress ONNX to FP16
│   ├── to_int8_synthetic() → Quantize with synthetic data
│   └── to_int8_calibrated() → Quantize with real data
├── CalibrationDataLoader
│   └── Load real frames for INT8 calibration
└── CLI utilities
    └── create_base_parser(), add_format_argument()
```

## Two Modes of Operation

**Embedded Mode** - Within bakery-luna
```python
from bakery_exporters import ExportPipeline
from bakery.catalog import config  # Uses catalog config

pipeline = ExportPipeline()
```

**Standalone Mode** - Independent package
```python
from bakery_exporters import ExportPipeline, config  # Has own config

config.models_dir = "/custom/path"
pipeline = ExportPipeline()
```

See [Configuration Guide](/.docs/CONFIG.md) for details.

## Documentation

- **[Configuration Guide](/.docs/CONFIG.md)** - Embedded vs Standalone
- **[Specification](/.docs/specs/exporters-specification.md)** - Complete API reference
- **[Architecture ADR](/.docs/adrs/001-exporters-architecture.md)** - Design decisions
- **[Usage Guide](./USAGE.md)** - Practical examples and workflows
- **[C4 Model](/.docs/C4-model.md)** - System architecture diagrams

## Core Modules

| Module | Responsibility |
|--------|-----------------|
| `onnx.py` | YOLO to ONNX export (OnnxExporter) |
| `openvino.py` | ONNX to OpenVINO IR conversion (OpenVINOConverter) |
| `calibration.py` | INT8 calibration data loading (CalibrationDataLoader) |
| `pipeline.py` | Export workflow orchestration (ExportPipeline) |
| `cli.py` | Shared CLI utilities and argument parsing |

## Common Workflows

### GPU Deployment (FP16)
```bash
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n s m --resolution 256 320 --format fp16
```

### CPU Deployment (INT8 - Fast)
```bash
uv run scripts/tasks/export/export_int8.py \
  --model 26 --model-size n --type segmentation --resolution 320 --format int8
```

### CPU Deployment (INT8 - Accurate)
```bash
# Step 1: Extract calibration data
uv run extract_calibration_frames.py --resolution 320

# Step 2: Export ONNX
uv run scripts/tasks/export/export_int8.py \
  --model 26 --model-size n --type segmentation --resolution 320 --format onnx

# Step 3: Calibrate with real data
uv run scripts/tasks/export/calibrate_int8.py \
  --model yolo26n-seg --resolution 320
```

## Dependencies

| Dependency | Size | Required For | Install |
|------------|------|--------------|---------|
| **ultralytics** | ~500MB | ONNX export | `[onnx]` |
| **openvino** | ~200MB | IR conversion | `[openvino]` |
| **nncf** | ~100MB | INT8 quantization | `[nncf]` |

## Performance

**Typical Export Times:**
- ONNX: 40-60s per model
- FP16: 5-10s per model
- INT8 (synthetic): 20-30s per model
- INT8 (calibrated): 60-120s per model

**Model Size Reduction:**
- FP16: 50-55% of original
- INT8: 25-30% of original

## Project Structure

```
bakery-exporters/
├── .docs/
│   ├── specs/
│   │   └── exporters-specification.md
│   ├── adrs/
│   │   └── 001-exporters-architecture.md
│   └── C4-model.md
├── src/bakery_exporters/
│   ├── __init__.py
│   ├── onnx.py
│   ├── openvino.py
│   ├── calibration.py
│   ├── pipeline.py
│   └── cli.py
├── tests/
│   └── __init__.py
├── USAGE.md
├── README.md
└── pyproject.toml
```

## Quick Examples

**Export via Python:**
```python
from bakery_exporters import ExportPipeline, ExportFormat

pipeline = ExportPipeline()
results = pipeline.export("26", "n", "segmentation", 320, [ExportFormat.FP16])
print(results[0].output_path)  # models/yolo26n-seg/320/fp16/model.xml
```

**Check available models:**
```python
from bakery.catalog import ModelPath

models = ModelPath.list_models(task="segmentation")
for m in models:
    print(f"{m['name']} @ {m['resolution']}px: {m['formats']}")
```

## Related Projects

- **[Bakery Catalog](/bakery/catalog/)** - Model discovery and configuration
- **[Bakery Luna](/bakery-luna/)** - Main vision pipeline package
- **[Scripts](/../scripts/tasks/export/)** - Export script implementations
