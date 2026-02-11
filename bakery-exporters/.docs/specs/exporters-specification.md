# Bakery Exporters Specification

**Version:** 0.1.0
**Last Updated:** 2025-02-10
**Related ADR:** [ADR-001: Exporters Architecture](/docs/adrs/001-exporters-architecture.md)

## Overview

Bakery Exporters is a modular system for exporting and converting YOLO models to optimized inference formats. It provides:

- **ONNX Export**: Convert YOLO models to ONNX format
- **OpenVINO Conversion**: Convert ONNX to OpenVINO IR (FP16, INT8)
- **INT8 Quantization**: Two strategies (synthetic for speed, calibrated for accuracy)
- **Model Pipeline**: Orchestrates complete export workflow
- **CLI Interface**: Consistent command-line tools

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│ Export Pipeline (CLI)                                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ ExportPipeline (Orchestrator)                        │  │
│  │  - Manages complete export workflow                  │  │
│  │  - Coordinates ONNX → OpenVINO conversions          │  │
│  │  - Handles errors and logging                        │  │
│  └──────────────────────────────────────────────────────┘  │
│           ↓                    ↓                    ↓        │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────┐   │
│  │ OnnxExporter │  │OpenVINOConverter │  │Calibration │   │
│  │              │  │                  │  │DataLoader  │   │
│  │ • export()   │  │ • to_fp16()      │  │            │   │
│  │              │  │ • to_int8_syn()  │  │ • __iter__ │   │
│  │              │  │ • to_int8_cal()  │  │ • __len__  │   │
│  └──────────────┘  └──────────────────┘  └────────────┘   │
│           ↓                    ↓                    ↓        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ bakery.catalog (Config, Paths)                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | Responsibility | Key Classes |
|--------|-----------------|-------------|
| `onnx.py` | YOLO to ONNX export | `OnnxExporter` |
| `openvino.py` | ONNX to OpenVINO IR conversion | `OpenVINOConverter`, `OpenVINOPrecision` |
| `calibration.py` | INT8 calibration data management | `CalibrationDataLoader`, `discover_onnx_models()` |
| `pipeline.py` | Export workflow orchestration | `ExportPipeline`, `ExportFormat`, `ExportResult` |
| `cli.py` | Command-line interface utilities | `create_base_parser()`, `add_format_argument()` |

## Export Formats

### Supported Formats

| Format | Extension | Use Case | Precision | Device |
|--------|-----------|----------|-----------|--------|
| ONNX | `.onnx` | Intermediate format | FP32 | CPU/GPU |
| FP16 | `.xml` + `.bin` | GPU inference | FP16 | GPU |
| INT8 | `.xml` + `.bin` | CPU inference (fast) | INT8 | CPU/VNNI |
| INT8_CALIBRATED | `.xml` + `.bin` | CPU inference (accurate) | INT8 | CPU/VNNI |

### Format Characteristics

#### ONNX (FP32)
- **When to use**: Intermediate format, model sharing, framework-agnostic
- **Size**: Original size
- **Precision**: Full (FP32)
- **Export time**: ~30-60 seconds
- **File structure**: Single `.onnx` file

#### FP16 (OpenVINO)
- **When to use**: GPU inference with reduced memory footprint
- **Size**: ~50% of original
- **Precision**: Half (FP16)
- **Conversion time**: ~5-10 seconds
- **File structure**: `model.xml` + `model.bin`
- **Requirements**: OpenVINO 2024.0+

#### INT8 (Synthetic)
- **When to use**: CPU inference, quick quantization, prototyping
- **Size**: ~25% of original
- **Precision**: Integer (INT8)
- **Quantization time**: ~20-30 seconds
- **Calibration data**: 100 random synthetic frames
- **File structure**: `model.xml` + `model.bin`
- **Requirements**: OpenVINO 2024.0+, NNCF 2.12+

#### INT8 (Calibrated)
- **When to use**: CPU inference with production accuracy
- **Size**: ~25% of original
- **Precision**: Integer (INT8)
- **Calibration time**: 1-2 minutes (depends on dataset size)
- **Calibration data**: Real preprocessed frames (.npy)
- **File structure**: `model.xml` + `model.bin`
- **Requirements**: OpenVINO 2024.0+, NNCF 2.12+, calibration data

## Data Flow

### Export Workflow

```
Input: (yolo_version, size, task, resolution, formats)
│
├─→ Step 1: Export ONNX
│   ├─ Load YOLO model: YOLO(model_name)
│   ├─ Export to ONNX: model.export(format="onnx", imgsz=resolution)
│   └─ Output: models/{model_name}/{resolution}/onnx/model.onnx
│
├─→ Step 2a: Convert to FP16 (if requested)
│   ├─ Load ONNX: ov.Core().read_model(onnx_path)
│   ├─ Save with compression: ov.save_model(..., compress_to_fp16=True)
│   └─ Output: models/{model_name}/{resolution}/fp16/model.xml + model.bin
│
├─→ Step 2b: Quantize to INT8 (if requested)
│   ├─ Generate synthetic data: np.random.rand(1, 3, res, res) × 100
│   ├─ Quantize: nncf.quantize(model, Dataset(synthetic_data))
│   └─ Output: models/{model_name}/{resolution}/int8/model.xml + model.bin
│
└─→ Step 2c: Calibrate INT8 (if requested)
    ├─ Load calibration data: CalibrationDataLoader(resolution)
    ├─ Quantize: nncf.quantize(model, Dataset(calibration_data), preset=MIXED)
    └─ Output: models/{model_name}/{resolution}/int8_calibrated/model.xml + model.bin

Output: List[ExportResult] with paths and success status
```

### Calibration Data Flow

```
Real video frames
│
├─→ extract_calibration_frames.py
│   └─ Preprocess to (1, 3, resolution, resolution)
│   └─ Save as .npy files
│
├─→ {BAKERY_CALIBRATION_DIR}/preprocessed_{resolution}/
│   ├─ frame_0000.npy
│   ├─ frame_0001.npy
│   └─ ...
│
└─→ CalibrationDataLoader(resolution)
    └─ Iterator[np.ndarray] for NNCF quantization
```

## Python API

### OnnxExporter

```python
from bakery_exporters import OnnxExporter

exporter = OnnxExporter(opset=17)  # opset optional, uses config.onnx_opset

# Export YOLO model to ONNX
onnx_path = exporter.export(
    model_name="yolo26n-seg",
    resolution=320,
    output_dir=None,  # uses ModelPath.build() if None
    simplify=True,
    dynamic=False
)
# Returns: Path to model.onnx
```

### OpenVINOConverter

```python
from bakery_exporters import OpenVINOConverter

converter = OpenVINOConverter()

# Convert to FP16
fp16_path = converter.to_fp16(onnx_path, "yolo26n-seg", 320)

# Quantize with synthetic data (fast)
int8_path = converter.to_int8_synthetic(
    onnx_path, "yolo26n-seg", 320,
    num_samples=100
)

# Calibrate with real data (accurate)
from bakery_exporters import CalibrationDataLoader
loader = CalibrationDataLoader(320)
int8_cal_path = converter.to_int8_calibrated(
    onnx_path, "yolo26n-seg", 320,
    calibration_data=loader,
    preset="MIXED"  # or "PERFORMANCE"
)
```

### CalibrationDataLoader

```python
from bakery_exporters import CalibrationDataLoader

loader = CalibrationDataLoader(resolution=320)

# Use with NNCF
import nncf
dataset = nncf.Dataset(loader)
```

### ExportPipeline

```python
from bakery_exporters import ExportPipeline, ExportFormat

pipeline = ExportPipeline()

# Export single model
results = pipeline.export(
    yolo_version="26",
    size="n",
    task="segmentation",
    resolution=320,
    formats=[ExportFormat.ONNX, ExportFormat.FP16],
    dry_run=False
)

# Export batch
all_results = pipeline.export_batch(
    yolo_version="26",
    sizes=["n", "s", "m"],
    task="segmentation",
    resolutions=[256, 320],
    formats=[ExportFormat.FP16, ExportFormat.INT8],
    dry_run=False
)

# ExportResult namedtuple
for result in results:
    print(f"{result.model_name} @ {result.resolution}px: {result.format.value}")
    print(f"  Success: {result.success}")
    if not result.success:
        print(f"  Error: {result.error}")
```

## Dependencies

### Required

None (unless using specific exporters)

### Optional

| Group | Purpose | Size | Includes |
|-------|---------|------|----------|
| `[onnx]` | YOLO export | ~500MB | ultralytics |
| `[openvino]` | IR conversion | ~200MB | openvino, openvino-dev |
| `[nncf]` | INT8 quantization | ~100MB | nncf |
| `[full]` | Everything | ~800MB | All above |

## Performance Metrics

### Typical Export Times

| Format | Model | Resolution | Time |
|--------|-------|------------|------|
| ONNX | yolo26n-seg | 320×320 | 40-60s |
| FP16 | yolo26n-seg | 320×320 | 5-10s |
| INT8 (synthetic) | yolo26n-seg | 320×320 | 20-30s |
| INT8 (calibrated) | yolo26n-seg | 320×320 | 60-120s |

### Model Size Reduction

| Format | yolo26n-seg | yolo26m-seg | yolo26l-seg |
|--------|-----------|-----------|-----------|
| ONNX (FP32) | 100% | 100% | 100% |
| FP16 | 50-55% | 50-55% | 50-55% |
| INT8 | 25-30% | 25-30% | 25-30% |

## Related

- **ADR:** [ADR-001: Exporters Architecture](/docs/adrs/001-exporters-architecture.md)
- **Usage:** [USAGE.md](/USAGE.md)
- **Scripts:** [../../../scripts/tasks/export/](/scripts/tasks/export/)
