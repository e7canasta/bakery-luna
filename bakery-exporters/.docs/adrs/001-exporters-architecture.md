# ADR-001: Exporters Architecture

**Status:** Accepted
**Date:** 2025-02-10
**Authors:** Claude, User

## Context

The export functionality in Bakery had several issues:

1. **Code Duplication**: 144+ lines of identical code across 4 scripts (export_to_onnx)
2. **Multiple Implementations**: 5 different OpenVINO conversion implementations doing similar tasks
3. **CLI Duplication**: ~80 lines of argument parsing duplicated in each script
4. **Heavy Dependencies**: ultralytics (500MB), openvino (200MB), nncf (100MB) required even for inference-only users
5. **Tight Coupling**: Export logic tightly coupled to scripts, hard to reuse

## Decision

Create `bakery-exporters` as a separate package with clear module responsibilities:

### Module Design

| Module | Responsibility |
|--------|-----------------|
| `onnx.py` | Consolidate all ONNX export logic (OnnxExporter class) |
| `openvino.py` | Unify OpenVINO conversions (OpenVINOConverter class) |
| `calibration.py` | Handle INT8 calibration data (CalibrationDataLoader) |
| `pipeline.py` | Orchestrate export workflows (ExportPipeline) |
| `cli.py` | Shared CLI utilities (create_base_parser, add_format_argument) |

### Dependency Management

**Optional Dependencies**:
- `[onnx]`: ultralytics
- `[openvino]`: openvino, openvino-dev
- `[nncf]`: nncf
- `[full]`: all above

Users only install what they need.

### Export Pipeline

```
1. OnnxExporter.export()
   └─ YOLO model → ONNX file

2. ExportPipeline routes to:
   ├─ OpenVINOConverter.to_fp16()       (GPU inference)
   ├─ OpenVINOConverter.to_int8_synthetic() (CPU prototype)
   └─ OpenVINOConverter.to_int8_calibrated() (CPU production)

3. CalibrationDataLoader
   └─ Iterator for NNCF quantization
```

## Consequences

### Positive

✅ **Code Consolidation**
- ONNX export: 1 implementation (vs 4 before)
- OpenVINO conversion: 1 implementation (vs 5 before)
- CLI args: 1 definition (vs 5 before)

✅ **Reduced Duplication**
- 1,004 lines removed (71% reduction in export scripts)
- Single source of truth for each operation

✅ **Reusable Components**
- OnnxExporter can be used standalone
- OpenVINOConverter is format-agnostic
- CalibrationDataLoader is compatible with any NNCF-based quantization

✅ **Optional Dependencies**
- bakery-luna core: only psutil, supervision
- Export tools: optional via bakery-exporters
- Users choose what to install

✅ **Clear API**
- ExportPipeline.export() for programmatic use
- CLI scripts as thin wrappers
- ExportFormat enum for type safety

### Negative

⚠️ **Additional Package to Maintain**
- bakery-exporters is separate package
- Must coordinate updates with bakery-luna

⚠️ **Potential Dependency Conflicts**
- ultralytics, openvino, nncf might have conflicting sub-deps
- Must manage version constraints carefully

## Alternatives Considered

### Alternative 1: Keep Monolithic Export Code

Maintain duplicated code in scripts.

**Rejected because:**
- 1,000+ lines of duplication hard to maintain
- Bug fixes require changes in 5 places
- No code reusability

### Alternative 2: Single Export Module Inside bakery

Put all export code inside bakery package with optional deps.

**Rejected because:**
- Makes bakery package larger
- Export concerns are separate from inference
- Harder to iterate on export features

### Alternative 3: Plugin System for Exporters

Allow third-party exporters to register formats.

**Rejected because:**
- Over-engineered for current needs
- Only 4 formats supported
- Simpler to add new formats to OpenVINOConverter

## Implementation

### File Structure

```
bakery-exporters/
├── pyproject.toml
├── README.md
├── USAGE.md
├── .docs/
│   ├── specs/
│   │   └── exporters-specification.md
│   └── adrs/
│       └── 001-exporters-architecture.md
└── src/bakery_exporters/
    ├── __init__.py
    ├── onnx.py
    ├── openvino.py
    ├── calibration.py
    ├── pipeline.py
    └── cli.py
```

### Export Script Refactoring

Before (343 lines):
```python
def export_to_onnx(...): ...  # 35 lines
def convert_to_fp16(...): ...  # 30 lines
def export_model(...): ...     # 60 lines
def main(): ...               # 215 lines
```

After (89 lines):
```python
pipeline = ExportPipeline()
all_results = pipeline.export_batch(...)  # 20 lines
```

### API Usage Examples

**Programmatic:**
```python
from bakery_exporters import ExportPipeline, ExportFormat

pipeline = ExportPipeline()
results = pipeline.export(
    "26", "n", "segmentation", 320,
    [ExportFormat.FP16]
)
```

**CLI:**
```bash
uv run export_sauron_segmentation.py \
  --model 26 --model-size n --resolution 320 --format fp16
```

## Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Export script lines | 1,414 | 410 | -71% |
| Code duplication | 144+ lines | 0 | -100% |
| ONNX implementations | 4 | 1 | -75% |
| OpenVINO implementations | 5 | 1 | -80% |
| bakery-luna core deps | 6 heavy | 2 light | -67% |

## Testing

✅ **Implemented:**
- OnnxExporter instantiation
- OpenVINOConverter methods
- ExportPipeline.export()
- CLI parser
- Script dry-runs

⏳ **Future:**
- Unit tests for each converter
- Integration tests with real models
- Performance benchmarks
- Calibration data tests

## Open Questions

1. Should we support other frameworks (TensorFlow, PyTorch) in the future?
   - Current design supports it (abstraction in pipeline)
   - No immediate need

2. Should export formats be pluggable?
   - Current hardcoded formats (ONNX, FP16, INT8) sufficient
   - Easy to add new formats to OpenVINOConverter

3. Should we support exporting to other runtimes (TensorRT, CoreML)?
   - Beyond scope of Bakery exporters
   - Users can wrap OnnxExporter output with other tools

## References

- Bakery Catalog ADR: [ADR-006 in bakery-luna](/bakery-luna/.docs/adrs/006-catalog-exporters-separation.md)
- Export Specification: [Exporters Spec](/docs/specs/exporters-specification.md)
- Usage Guide: [USAGE.md](/USAGE.md)
