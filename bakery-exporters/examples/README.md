# Bakery Exporters Examples

Practical examples demonstrating how to use the Bakery Exporters package.

## Examples

### 1. Basic Export (`01_basic_export.py`)

Simple ONNX export of a YOLO model.

```bash
python examples/01_basic_export.py
```

**What it does:**
- Creates an OnnxExporter
- Exports yolo26n-seg to ONNX format
- Shows output path

**Key concepts:**
- OnnxExporter class
- ModelPath usage
- Config integration

---

### 2. Export Multiple Formats (`02_export_multiple_formats.py`)

Export a single model to multiple formats (ONNX, FP16, INT8).

```bash
python examples/02_export_multiple_formats.py
```

**What it does:**
- Creates an ExportPipeline
- Exports to ONNX, FP16, and INT8 in one pipeline
- Shows results for each format

**Key concepts:**
- ExportPipeline orchestration
- ExportFormat enum
- Pipeline results handling

---

### 3. Batch Export (`03_batch_export.py`)

Export multiple models and resolutions in a single batch.

```bash
python examples/03_batch_export.py
```

**What it does:**
- Exports 2 sizes (n, s) × 2 resolutions (256, 320) = 4 models
- Each model → 2 formats (FP16, INT8) = 8 total exports
- Shows summary by model

**Key concepts:**
- Batch processing with export_batch()
- Managing multiple export configurations
- Result aggregation and summary

---

### 4. INT8 Calibration (`04_int8_calibration.py`)

Export and calibrate a model to INT8 using real calibration data.

```bash
# First, prepare calibration data
python extract_calibration_frames.py --resolution 320

# Then run the example
python examples/04_int8_calibration.py
```

**What it does:**
- Checks if calibration data exists
- Exports model to ONNX
- Calibrates INT8 with real data (1-2 minutes)
- Shows both ONNX and calibrated INT8 paths

**Key concepts:**
- CalibrationDataLoader
- Real data calibration vs synthetic
- Step-by-step pipeline
- Error handling and user guidance

---

### 5. Configuration Modes (`05_embedded_vs_standalone.py`)

Demonstrates the two configuration modes of bakery-exporters.

```bash
python examples/05_embedded_vs_standalone.py
```

**What it does:**
- Shows embedded mode (using bakery.catalog config)
- Shows standalone mode (using bakery_exporters config)
- Demonstrates config conversion
- Shows environment variable priorities

**Key concepts:**
- Embedded vs standalone usage
- Configuration hierarchy
- Runtime config overrides
- Config interoperability

**When to use:**
- Understanding how to use bakery-exporters in different contexts
- Learning configuration best practices
- Debugging configuration issues

---

## Running Examples

### From bakery-exporters directory

```bash
cd bakery-exporters

# Example 1
python examples/01_basic_export.py

# Example 2
python examples/02_export_multiple_formats.py

# Example 3
python examples/03_batch_export.py

# Example 4
python examples/04_int8_calibration.py
```

### From project root

```bash
python bakery-exporters/examples/01_basic_export.py
```

## Configuration

Examples use environment variables from `.env` file or defaults:

```bash
BAKERY_MODELS_DIR=models
BAKERY_CALIBRATION_DIR=calibration_data
BAKERY_DEFAULT_YOLO=11
BAKERY_ONNX_OPSET=17
```

Or set at runtime:

```bash
export BAKERY_MODELS_DIR=/custom/models
python examples/01_basic_export.py
```

## Requirements

Install with full dependencies:

```bash
pip install bakery-luna bakery-exporters[full]
```

Or selective dependencies:

```bash
# Just ONNX
pip install bakery-luna bakery-exporters[onnx]

# Just OpenVINO
pip install bakery-luna bakery-exporters[openvino]

# Just NNCF
pip install bakery-luna bakery-exporters[nncf]
```

## Output

Examples create models in the configured `BAKERY_MODELS_DIR`:

```
models/
├── yolo26n-seg/
│   ├── 256/
│   │   ├── onnx/model.onnx
│   │   ├── fp16/model.xml + model.bin
│   │   ├── int8/model.xml + model.bin
│   │   └── int8_calibrated/model.xml + model.bin
│   └── 320/
│       └── ...
└── yolo26s-seg/
    └── ...
```

## Troubleshooting

### ImportError: ultralytics is required

Install ONNX support:
```bash
pip install bakery-exporters[onnx]
```

### FileNotFoundError: Calibration data not found

Prepare calibration data first:
```bash
python extract_calibration_frames.py --resolution 320
```

### OutOfMemoryError during quantization

- Use smaller model (yolo26n vs yolo26l)
- Reduce number of calibration frames
- Use synthetic quantization instead

## Next Steps

After running examples:

1. **Customize exports** - Modify example scripts for your models
2. **Batch processing** - Use example 3 pattern for production workflows
3. **Integration** - Embed export logic into your own tools
4. **Advanced usage** - See [USAGE.md](../USAGE.md) for more details

## Related Documentation

- [USAGE.md](../USAGE.md) - Complete usage guide
- [Specification](../.docs/specs/exporters-specification.md) - API reference
- [Architecture ADR](../.docs/adrs/001-exporters-architecture.md) - Design decisions
