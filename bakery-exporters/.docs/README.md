# Bakery Exporters Documentation

## Overview

This directory contains comprehensive documentation for the Bakery Exporters package.

## Documentation Structure

### 0. **Configuration** (`CONFIG.md`)
Comprehensive guide to configuration in both embedded and standalone modes.

- **[CONFIG.md](CONFIG.md)**
  - Embedded mode (within bakery-luna)
  - Standalone mode (independent package)
  - Environment variables reference
  - Configuration API
  - Integration patterns
  - Common scenarios (dev, prod, Docker)
  - Troubleshooting

### 1. **Specification** (`specs/`)
Complete technical specification of Bakery Exporters functionality.

- **[exporters-specification.md](specs/exporters-specification.md)**
  - Overview and architecture
  - Supported export formats and their characteristics
  - Data flow diagrams
  - Complete Python API reference
  - CLI reference
  - Dependencies and installation
  - Performance metrics

### 2. **Architecture Decisions** (`adrs/`)
Architecture Decision Records documenting design choices and rationale.

- **[001-exporters-architecture.md](adrs/001-exporters-architecture.md)**
  - Context: problems with previous export implementation
  - Decision: create separate bakery-exporters package with modular design
  - Consequences: code consolidation, reduced duplication, optional dependencies
  - Alternatives considered and rejected
  - Implementation details and migration path
  - Metrics showing improvements (71% code reduction, -100% duplication)

### 3. **Usage Guide** (`../USAGE.md`)
Practical examples and common workflows for end users.

- Quick start examples
- Format selection guide (ONNX, FP16, INT8)
- Common workflows:
  - GPU deployment (FP16)
  - CPU deployment (INT8 quick)
  - CPU deployment (INT8 accurate)
  - All formats at once
- Advanced usage (batch export, custom calibration)
- Troubleshooting guide
- Environment variables reference
- Performance tips
- Detailed examples for different scenarios

### 4. **System Architecture** (`C4-model.md`)
C4 model diagrams showing system architecture at different levels of abstraction.

- **Level 1 (Context)**: Bakery Exporters in broader ecosystem
- **Level 2 (Containers)**: CLI scripts, Python package, documentation, external dependencies
- **Level 3 (Components)**: Internal modules and their relationships
- **Data Flow Diagram**: Export workflow and calibration data flow
- **Deployment Context**: Different deployment scenarios (GPU, CPU, edge)
- **Dependencies & Integration**: Dependency tree and optional installation groups
- **Version 0.1.0 Components**: Complete list of v0.1.0 components

## Quick Navigation

### For End Users
1. Start with **[USAGE.md](../USAGE.md)** for practical examples
2. Refer to **[Specification](specs/exporters-specification.md)** for API details
3. Use **[C4 Model](C4-model.md)** to understand system architecture

### For Developers
1. Read **[Architecture ADR](adrs/001-exporters-architecture.md)** for design rationale
2. Study **[Specification](specs/exporters-specification.md)** for implementation details
3. Review **[C4 Model](C4-model.md)** for system design

### For DevOps/Deployment
1. Check **[USAGE.md](../USAGE.md)** deployment scenarios section
2. Review **[C4 Model](C4-model.md)** deployment context
3. Use **[Specification](specs/exporters-specification.md)** performance metrics

## Key Concepts

### Export Formats Hierarchy

```
ONNX (FP32) ─┬─→ FP16 (GPU)
             ├─→ INT8 (Synthetic) → Quick, ~95% accuracy
             └─→ INT8 (Calibrated) → Accurate, ~98% accuracy
```

### Module Responsibilities

| Module | Responsibility | Key Classes |
|--------|-----------------|-------------|
| `onnx.py` | YOLO to ONNX export | `OnnxExporter` |
| `openvino.py` | ONNX to IR conversion | `OpenVINOConverter` |
| `calibration.py` | INT8 calibration data | `CalibrationDataLoader` |
| `pipeline.py` | Export orchestration | `ExportPipeline` |
| `cli.py` | CLI utilities | `create_base_parser()` |

### Dependency Strategy

- **Core (always)**: bakery-luna (for catalog/config)
- **Optional [onnx]**: ultralytics (~500MB)
- **Optional [openvino]**: openvino (~200MB)
- **Optional [nncf]**: nncf (~100MB)
- **Combined [full]**: ~800MB total

Users install only what they need.

## Performance Summary

### Export Time by Format

| Format | Model | Time |
|--------|-------|------|
| ONNX | yolo26n-seg | 40-60s |
| FP16 | yolo26n-seg | 5-10s |
| INT8 (synthetic) | yolo26n-seg | 20-30s |
| INT8 (calibrated) | yolo26n-seg | 60-120s |

### Model Size Reduction

- FP16: 50-55% of original
- INT8: 25-30% of original

## Version Information

**Current Version:** 0.1.0
**Last Updated:** 2025-02-10
**Status:** Accepted

## File Index

```
bakery-exporters/.docs/
├── README.md (this file)
├── specs/
│   └── exporters-specification.md
│       ├── Overview and architecture
│       ├── Supported export formats
│       ├── Data flow diagrams
│       ├── Python API reference
│       ├── CLI reference
│       ├── Dependencies
│       └── Performance metrics
├── adrs/
│   └── 001-exporters-architecture.md
│       ├── Context and decision
│       ├── Consequences (positive/negative)
│       ├── Alternatives considered
│       ├── Implementation details
│       ├── Metrics
│       └── References
├── C4-model.md
│   ├── Context diagram (Level 1)
│   ├── Container diagram (Level 2)
│   ├── Component diagram (Level 3)
│   ├── Data flow diagram
│   ├── Deployment context
│   └── Dependencies & integration
└── (../USAGE.md - in parent directory)
    ├── Quick start
    ├── Installation options
    ├── Python API examples
    ├── CLI usage examples
    ├── Format selection guide
    ├── Common workflows
    ├── Advanced usage
    ├── Troubleshooting
    ├── Environment variables
    ├── Performance tips
    └── Detailed examples
```

## Related Documentation

- **Bakery Luna**: [/bakery-luna/](../bakery-luna/)
  - Main vision pipeline package
  - [ADR-006: Catalog & Exporters Separation](../bakery-luna/.docs/adrs/006-catalog-exporters-separation.md)

- **Bakery Catalog**: [/bakery/catalog/](../bakery/catalog/)
  - Model discovery and configuration
  - [Model Catalog Specification](../bakery-luna/.docs/specs/model-catalog.md)
  - [ADR-005: Model Catalog Configuration](../bakery-luna/.docs/adrs/005-model-catalog-configuration.md)

## Getting Help

### Common Questions

**Q: Which format should I use?**
- A: See "Format Selection Guide" in [USAGE.md](../USAGE.md)

**Q: How do I export a model?**
- A: See "Quick Start" section in [USAGE.md](../USAGE.md)

**Q: What are the dependencies?**
- A: See "Dependencies" section in [exporters-specification.md](specs/exporters-specification.md)

**Q: Why is this a separate package?**
- A: See [ADR-001: Exporters Architecture](adrs/001-exporters-architecture.md)

### Troubleshooting

See [USAGE.md](../USAGE.md) "Troubleshooting" section for common errors and solutions.

## Contributing

To update documentation:

1. Edit relevant markdown files in `.docs/`
2. Update version and "Last Updated" date
3. Ensure consistency across all documentation files
4. Test any examples provided

## License

Part of Bakery Luna project.
