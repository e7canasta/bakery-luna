# Bakery Exporters - CLI Scripts

Command-line tools for exporting and calibrating YOLO models.

## Available Commands

### bakery-export-segmentation
Export YOLO segmentation models to optimized formats.

```bash
# Single model to FP16 (GPU)
bakery-export-segmentation --model 26 --model-size n --resolution 320

# Multiple models to multiple formats
bakery-export-segmentation --model 26 --model-size n s m --resolution 256 320 --format onnx fp16

# All combinations (dry-run)
bakery-export-segmentation --model 26 --all --dry-run
```

### bakery-export-pose
Export YOLO pose estimation models to optimized formats.

```bash
# Export pose model
bakery-export-pose --model 26 --model-size m --resolution 256

# Multiple resolutions
bakery-export-pose --model 26 --model-size n s m --resolution 256 320
```

### bakery-export-int8
Export YOLO models to INT8 format for CPU inference.

```bash
# Quick INT8 with synthetic calibration
bakery-export-int8 --model 26 --model-size n --type segmentation --resolution 320

# Export with ONNX
bakery-export-int8 --model 26 --model-size n --type segmentation --resolution 320 --format onnx int8

# Pose estimation
bakery-export-int8 --model 26 --model-size m --type pose --resolution 256
```

### bakery-calibrate-int8
Calibrate INT8 models using real calibration data for better accuracy.

```bash
# Calibrate specific model
bakery-calibrate-int8 --model yolo26n-seg --resolution 320

# Calibrate all available models
bakery-calibrate-int8 --all

# Calibrate at specific resolution
bakery-calibrate-int8 --all --resolution 320
```

## Common Arguments

All export commands support:

```
--model, -m         YOLO version (8, 11, 26) [default: 11]
--model-size, -s    Model size(s) (n, s, m, l, x)
--type, -t          Task type (detection, segmentation, pose) [segmentation tasks only]
--resolution, -r    Resolution(s) (160-640)
--format, -f        Format(s) to export (onnx, fp16, int8, int8_calibrated)
--all               Export all combinations
--dry-run           Show plan without executing
--output, -o        Override models directory
```

## Installation

The scripts are installed with the bakery-exporters package:

```bash
pip install bakery-luna bakery-exporters[full]
```

Or selectively:

```bash
pip install bakery-luna bakery-exporters[onnx]      # For segmentation/pose
pip install bakery-luna bakery-exporters[openvino]  # For conversion
pip install bakery-luna bakery-exporters[nncf]      # For INT8 calibration
```

## Configuration

Scripts use environment variables:

```bash
export BAKERY_MODELS_DIR=models
export BAKERY_CALIBRATION_DIR=calibration_data
export BAKERY_DEFAULT_YOLO=11
export BAKERY_ONNX_OPSET=17
```

Or `.env` file:
```
BAKERY_MODELS_DIR=models
BAKERY_CALIBRATION_DIR=calibration_data
BAKERY_DEFAULT_YOLO=11
BAKERY_ONNX_OPSET=17
```

## Common Workflows

### GPU Deployment (FP16)

```bash
# Export segmentation models to FP16
bakery-export-segmentation --model 26 --model-size n s m --resolution 256 320

# Export pose models to FP16
bakery-export-pose --model 26 --model-size m l --resolution 256
```

### CPU Deployment (INT8 - Quick)

```bash
bakery-export-int8 \
  --model 26 \
  --model-size n s m \
  --type segmentation \
  --resolution 256 320
```

### CPU Deployment (INT8 - Accurate)

```bash
# Step 1: Prepare calibration data
# (Use your own data extraction tool)

# Step 2: Export ONNX
bakery-export-int8 \
  --model 26 \
  --model-size n \
  --type segmentation \
  --resolution 320 \
  --format onnx

# Step 3: Calibrate with real data
bakery-calibrate-int8 --model yolo26n-seg --resolution 320
```

## Troubleshooting

### Command not found

Install with bash entry points:
```bash
pip install bakery-exporters[full]

# Or manually run Python script
python -m bakery_exporters.scripts.bakery-export-segmentation --help
```

### ImportError: ultralytics

Install ONNX support:
```bash
pip install bakery-exporters[onnx]
```

### FileNotFoundError: Calibration data

Prepare calibration data first (using your data extraction tool)

## Related Documentation

- [USAGE.md](../USAGE.md) - Complete usage guide
- [Examples](../examples/) - Code examples
- [Specification](../.docs/specs/exporters-specification.md) - API reference
