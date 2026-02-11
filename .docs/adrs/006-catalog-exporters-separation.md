# ADR-006: Separation of Catalog and Exporters into Bounded Contexts

**Status:** Accepted
**Date:** 2025-02-10
**Authors:** Claude, User

## Context

The Bakery Luna package had grown to contain mixed responsibilities:

1. **Model Catalog**: Path building, discovery, configuration (env-based)
2. **Export/Conversion**: ONNX export, OpenVINO conversion, INT8 quantization, calibration

Additionally, the export scripts (`export_sauron_segmentation.py`, `export_int8.py`, etc.) had significant code duplication:
- 144+ lines of duplicated ONNX export code
- 5 different OpenVINO conversion implementations
- ~80 lines of CLI argument parsing duplicated across 5 scripts

Heavy dependencies (ultralytics ~500MB, openvino ~200MB, nncf ~100MB) were installed even when not needed for inference.

## Decision

Separate Bakery into two bounded contexts with clear responsibilities:

### 1. **bakery/catalog/** (inside bakery package)

**Responsibility:** "Where are the models"

- Model path building and discovery
- Environment-based configuration
- Constants (YOLO versions, sizes, tasks, resolutions)
- No heavy external dependencies

**Modules:**
- `config.py`: BakeryConfig, environment loading
- `paths.py`: ModelPath class for path building
- `constants.py`: Domain constants
- `discovery.py`: Model listing and existence checking
- `__init__.py`: Public API

**Usage:**
```python
from bakery.catalog import config, ModelPath
path = ModelPath.get("yolo26n-seg", 320, "fp16")
```

### 2. **bakery-exporters/** (separate package)

**Responsibility:** "How to create/convert models"

- ONNX export from YOLO
- OpenVINO IR conversion (FP16, INT8)
- INT8 calibration with real data
- Export pipeline orchestration
- Shared CLI utilities

**Modules:**
- `onnx.py`: OnnxExporter (consolidates 4 scripts)
- `openvino.py`: OpenVINOConverter (consolidates 5 implementations)
- `calibration.py`: CalibrationDataLoader
- `pipeline.py`: ExportPipeline orchestrator
- `cli.py`: Shared CLI argument parsing

**Dependencies:** Optional, installable separately
- `[onnx]`: ultralytics
- `[openvino]`: openvino, openvino-dev
- `[nncf]`: nncf
- `[full]`: all of the above

**Usage:**
```python
from bakery_exporters import ExportPipeline, ExportFormat

pipeline = ExportPipeline()
results = pipeline.export(
    "26", "n", "segmentation", 320,
    [ExportFormat.FP16]
)
```

### 3. **Export Scripts Refactoring**

All export scripts in `scripts/tasks/export/` became thin wrappers around `bakery_exporters`:

**Before:**
- `export_sauron_segmentation.py`: 343 lines
- `export_sauron_pose.py`: 343 lines
- `export_int8.py`: 362 lines
- `calibrate_int8.py`: 366 lines
- Total: 1,414 lines, 144+ lines duplicated

**After:**
- `export_sauron_segmentation.py`: 89 lines
- `export_sauron_pose.py`: 88 lines
- `export_int8.py`: 76 lines
- `calibrate_int8.py`: 157 lines
- Total: 410 lines (71% reduction)

**Reduction:** 1,004 lines of duplicated code eliminated

## Consequences

### Positive

✅ **Single Source of Truth**
- ONNX export logic: once (OnnxExporter)
- OpenVINO conversions: once (OpenVINOConverter)
- CLI args: once (create_base_parser)

✅ **Lightweight Core Package**
- bakery-luna: ~2 light dependencies (psutil, supervision)
- Export tools: optional via bakery-exporters[full]
- Users doing inference don't install ultralytics/openvino/nncf

✅ **Clear Separation of Concerns**
- bakery.catalog: stable, rarely changes
- bakery_exporters: evolves with new formats/optimizations

✅ **Better Dependency Management**
- Optional dependencies: users choose what to install
- CI/CD can install minimal deps for testing

✅ **Maintainability**
- 1,004 fewer lines of code
- Bug fixes in export logic automatically apply to all scripts
- Adding new format only requires updating OpenVINOConverter

### Negative

⚠️ **Additional Package**
- bakery-exporters as separate package to maintain
- Must coordinate updates between packages

⚠️ **Installation Complexity**
- Users running export must install bakery-exporters[full]
- May cause confusion about which package to install

⚠️ **Possible Circular Imports**
- bakery-exporters depends on bakery.catalog
- Must be careful not to create circular dependencies

## Alternatives Considered

### Alternative 1: Single Package with Optional Dependencies

Keep everything in bakery-luna but make openvino/nncf/ultralytics optional.

**Rejected because:**
- Doesn't address code duplication in scripts
- Optional imports make testing harder
- Conceptual separation is still unclear

### Alternative 2: Move Everything to Separate Packages

Create bakery-catalog and bakery-exporters as completely separate packages.

**Rejected because:**
- catalog is core and should stay with main package
- catalog rarely changes, exporters frequently do
- Most users need catalog, only power users need exporters

### Alternative 3: Keep Monolithic Approach

Continue duplicating code across scripts.

**Rejected because:**
- Maintenance burden too high
- Bug fixes require changes in 5 places
- ~1,000 lines of unnecessary duplication

## Implementation Details

### Backwards Compatibility

✅ **Maintained via shim layer:**
```python
# Old import (with deprecation warning)
from bakery.config import config, ModelPath

# New import (recommended)
from bakery.catalog import config, ModelPath
```

The old `bakery.config` module now re-exports everything from `bakery.catalog` with a DeprecationWarning.

### Migration Path

1. ✅ Phase 1: Created bakery/catalog/ with all config logic
2. ✅ Phase 2: Created bakery-exporters/ package
3. ✅ Phase 3: Refactored export scripts to use bakery-exporters
4. ✅ Phase 4: Updated bakery-luna/pyproject.toml (removed heavy deps)
5. ⏳ Phase 5: Deprecate bakery.config (v0.2.0)
6. ⏳ Phase 6: Remove bakery.config shim (v1.0.0)

### Files Changed

**Bakery Package:**
- Created: `bakery/catalog/` (5 new files)
- Updated: `bakery/config.py` (now a shim)
- Updated: `bakery/__init__.py` (imports from catalog)
- Updated: `pyproject.toml` (removed heavy deps)

**Bakery-Exporters Package:**
- Created: `bakery-exporters/` (8 new files)
- Provides: OnnxExporter, OpenVINOConverter, CalibrationDataLoader, ExportPipeline

**Export Scripts:**
- Updated: `export_sauron_segmentation.py` (343→89 lines)
- Updated: `export_sauron_pose.py` (343→88 lines)
- Updated: `export_int8.py` (362→76 lines)
- Updated: `calibrate_int8.py` (366→157 lines)

## Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Bakery deps | 6 heavy | 2 light | -67% |
| Export code duplication | 144+ lines | 0 lines | -100% |
| Script avg lines | 354 lines | 102 lines | -71% |
| ONNX export impls | 4 | 1 | -75% |
| OpenVINO conv impls | 5 | 1 | -80% |

## Testing

✅ **Tested:**
- bakery.catalog imports work
- bakery.config deprecation warning shown
- bakery-exporters package imports
- CLI dry-run works
- All 4 refactored scripts work

⏳ **Still needed:**
- Unit tests for OpenVINOConverter
- Integration tests for full export pipeline
- Tests for CalibrationDataLoader with real data

## References

- **Previous ADR:** [ADR-005: Model Catalog Configuration](/docs/adrs/005-model-catalog-configuration.md)
- **Code:** [bakery/catalog/](/bakery/catalog/)
- **Code:** [bakery-exporters/](/bakery-exporters/)
- **Scripts:** [scripts/tasks/export/](/scripts/tasks/export/)
