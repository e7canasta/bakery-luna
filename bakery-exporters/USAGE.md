# Bakery Exporters - Usage Guide

## Installation

### Basic Setup

```bash
# Install bakery-luna (catalog + core)
pip install bakery-luna

# Install exporters with all dependencies
pip install bakery-luna bakery-exporters[full]
```

### Selective Installation

```bash
# For ONNX export only
pip install bakery-luna bakery-exporters[onnx]

# For OpenVINO conversion only
pip install bakery-luna bakery-exporters[openvino]

# For INT8 quantization only
pip install bakery-luna bakery-exporters[nncf]
```

## Quick Start

### Export via CLI

```bash
# 1. Export segmentation model to FP16
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n --resolution 320

# 2. Check what would be exported (dry-run)
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n --resolution 320 --dry-run

# 3. Export multiple sizes
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n s m --resolution 256 320
```

### Export via Python

```python
from bakery_exporters import ExportPipeline, ExportFormat

# Create pipeline
pipeline = ExportPipeline()

# Export single model
results = pipeline.export(
    yolo_version="26",
    size="n",
    task="segmentation",
    resolution=320,
    formats=[ExportFormat.FP16]
)

# Check results
for result in results:
    if result.success:
        print(f"✓ {result.output_path}")
    else:
        print(f"✗ {result.error}")
```

## Common Workflows

### 1. GPU Deployment (FP16)

For inference on GPU with reduced memory.

```bash
# Export segmentation and pose models to FP16
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n s m --resolution 256 320

uv run scripts/tasks/export/export_sauron_pose.py \
  --model 26 --model-size m l --resolution 256
```

**Result:**
```
models/
├── yolo26n-seg/320/fp16/model.xml + model.bin
├── yolo26s-seg/256/fp16/model.xml + model.bin
└── yolo26m-pose/256/fp16/model.xml + model.bin
```

### 2. CPU Deployment (INT8 - Quick)

For fast prototyping on CPU.

```bash
# Export to INT8 with synthetic calibration (1 minute total)
uv run scripts/tasks/export/export_int8.py \
  --model 26 --model-size n --type segmentation --resolution 320 --format int8
```

**Tradeoff:** Smaller model size, slightly lower accuracy

### 3. CPU Deployment (INT8 - Accurate)

For production CPU inference with best accuracy.

```bash
# Step 1: Extract real calibration data
uv run extract_calibration_frames.py --resolution 320

# Step 2: Export to ONNX
uv run scripts/tasks/export/export_int8.py \
  --model 26 --model-size n --type segmentation --resolution 320 --format onnx

# Step 3: Calibrate with real data
uv run scripts/tasks/export/calibrate_int8.py \
  --model yolo26n-seg --resolution 320
```

**Result:**
```
models/yolo26n-seg/320/int8_calibrated/model.xml + model.bin
```

### 4. All Formats at Once

Compare different formats on same model.

```bash
# Export to all formats
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n --resolution 320 \
  --format onnx fp16

# Then calibrate INT8
uv run scripts/tasks/export/export_int8.py \
  --model 26 --model-size n --type segmentation --resolution 320 \
  --format int8
```

## Format Selection Guide

### ONNX (FP32)
- **Use case**: Sharing models, framework-agnostic deployment
- **Size**: 100% (baseline)
- **Precision**: Full (FP32)
- **Export time**: 40-60s
- **Inference**: Slower than optimized formats

```bash
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n --resolution 320 --format onnx
```

### FP16 (GPU)
- **Use case**: GPU inference with memory constraints
- **Size**: 50% of ONNX
- **Precision**: Half (FP16)
- **Export time**: 5-10s (fast!)
- **Inference**: 2x faster than FP32 on GPU

```bash
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n --resolution 320 --format fp16
```

### INT8 Synthetic (CPU Quick)
- **Use case**: Rapid prototyping on CPU
- **Size**: 25% of ONNX (4x smaller!)
- **Precision**: Integer (INT8)
- **Export time**: 20-30s
- **Accuracy**: ~95% of original
- **Inference**: 4-6x faster than FP32

```bash
uv run scripts/tasks/export/export_int8.py \
  --model 26 --model-size n --type segmentation --resolution 320 --format int8
```

### INT8 Calibrated (CPU Production)
- **Use case**: Production CPU inference
- **Size**: 25% of ONNX (4x smaller!)
- **Precision**: Integer (INT8)
- **Calibration time**: 60-120s
- **Accuracy**: 98%+ of original
- **Inference**: 4-6x faster than FP32

```bash
# Need real calibration data first
uv run extract_calibration_frames.py --resolution 320

# Then calibrate
uv run scripts/tasks/export/calibrate_int8.py \
  --model yolo26n-seg --resolution 320
```

## Advanced Usage

### Batch Export with Python

```python
from bakery_exporters import ExportPipeline, ExportFormat

pipeline = ExportPipeline()

# Export many models at once
results = pipeline.export_batch(
    yolo_version="26",
    sizes=["n", "s", "m"],
    task="segmentation",
    resolutions=[256, 320, 640],
    formats=[ExportFormat.ONNX, ExportFormat.FP16, ExportFormat.INT8]
)

# Total: 9 models × 3 formats = 27 export operations
print(f"Exported: {sum(1 for r in results if r.success)}/{len(results)}")
```

### Custom ONNX Export

```python
from bakery_exporters import OnnxExporter

exporter = OnnxExporter(opset=17)

# Use different ONNX opset
onnx_path = exporter.export(
    model_name="yolo26n-seg",
    resolution=320,
    simplify=True,      # Simplify ONNX graph
    dynamic=False       # Use static shapes
)

print(f"Exported: {onnx_path}")
```

### Custom Calibration

```python
from bakery_exporters import OpenVINOConverter, CalibrationDataLoader
import nncf

converter = OpenVINOConverter()
loader = CalibrationDataLoader(320)

# Calibrate with custom preset
onnx_path = ...  # your ONNX file

int8_path = converter.to_int8_calibrated(
    onnx_path,
    "yolo26n-seg",
    320,
    calibration_data=loader,
    preset="PERFORMANCE"  # vs "MIXED"
)
```

### Check Available Models

```python
from bakery.catalog import ModelPath

# List all exported models
all_models = ModelPath.list_models()

# List only segmentation models
seg_models = ModelPath.list_models(task="segmentation")

# Check specific model
exists = ModelPath.exists("yolo26n-seg", 320, "fp16")
print(f"FP16 model exists: {exists}")

# Get path to model
path = ModelPath.get("yolo26n-seg", 320, "fp16")
print(f"Model path: {path}")
```

## Troubleshooting

### ImportError: ultralytics is required

```bash
pip install bakery-exporters[onnx]
```

### FileNotFoundError: ONNX not found

Check the ONNX export step completed:
1. Is the model name correct? (e.g., `yolo26n-seg`)
2. Do you have disk space?
3. Check permissions in models directory

### FileNotFoundError: Calibration data not found

Extract calibration frames first:
```bash
uv run extract_calibration_frames.py --resolution 320
```

### OutOfMemoryError during quantization

Reduce calibration data:
```bash
# Use fewer frames
uv run calibrate_int8.py --model yolo26n-seg --resolution 320
```

Or use synthetic quantization instead:
```bash
uv run export_int8.py \
  --model 26 --model-size n --type segmentation --resolution 320
```

## Environment Variables

```bash
# Set models directory
export BAKERY_MODELS_DIR=/path/to/models

# Set calibration directory
export BAKERY_CALIBRATION_DIR=/path/to/calibration_data

# Set default YOLO version
export BAKERY_DEFAULT_YOLO=11

# Set ONNX opset
export BAKERY_ONNX_OPSET=17
```

Or in `.env`:
```
BAKERY_MODELS_DIR=models
BAKERY_CALIBRATION_DIR=calibration_data
BAKERY_DEFAULT_YOLO=11
BAKERY_ONNX_OPSET=17
```

## Performance Tips

### Reduce Export Time

1. **Export only needed formats**
   - FP16 for GPU (5-10s)
   - INT8 synthetic for CPU (20-30s)

2. **Use smaller models**
   - yolo26n is fastest to export
   - yolo26x takes ~2x longer

3. **Batch exports**
   - Export multiple models in one command
   - Share intermediate ONNX files

### Reduce Model Size

| Format | Size | Speed-up | Notes |
|--------|------|----------|-------|
| ONNX | 100% | 1x | Baseline |
| FP16 | 50% | 2-3x | GPU only |
| INT8 | 25% | 4-6x | CPU, real calibration for accuracy |

## Examples

### Example 1: Export for GPU + CPU

```bash
# GPU models (FP16)
uv run scripts/tasks/export/export_sauron_segmentation.py \
  --model 26 --model-size n s --resolution 320 --format fp16

# CPU models (INT8)
uv run scripts/tasks/export/export_int8.py \
  --model 26 --model-size n s --type segmentation --resolution 320 --format int8
```

### Example 2: Production Workflow

```bash
# 1. Prepare calibration data
uv run extract_calibration_frames.py --resolution 320

# 2. Export ONNX
uv run scripts/tasks/export/export_int8.py \
  --model 26 --model-size n --type segmentation --resolution 320 --format onnx

# 3. Calibrate INT8
uv run scripts/tasks/export/calibrate_int8.py --model yolo26n-seg --resolution 320

# 4. Verify model exists
python -c "from bakery.catalog import ModelPath; print(ModelPath.exists('yolo26n-seg', 320, 'int8_calibrated'))"
```

### Example 3: Programmatic Export

```python
from bakery_exporters import ExportPipeline, ExportFormat
from bakery.catalog import ModelPath

pipeline = ExportPipeline()

# Export
results = pipeline.export(
    "26", "n", "segmentation", 320,
    [ExportFormat.FP16, ExportFormat.INT8]
)

# Use models
fp16_path = ModelPath.get("yolo26n-seg", 320, "fp16")
int8_path = ModelPath.get("yolo26n-seg", 320, "int8")

print(f"FP16: {fp16_path}")
print(f"INT8: {int8_path}")
```

## API Reference

See [exporters-specification.md](/.docs/specs/exporters-specification.md) for complete API documentation.

## See Also

- [Exporters Specification](/.docs/specs/exporters-specification.md)
- [Architecture ADR](/.docs/adrs/001-exporters-architecture.md)
- [Bakery Catalog](/../bakery/catalog/)
