# Agent Guidelines for Bakery

Bakery is a modular computer vision pipeline for dual-model inference (segmentation + pose estimation) optimized for OpenVINO runtime.

## Build/Test/Lint Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/test_geometry.py

# Run single test class
uv run pytest tests/test_geometry.py::TestNMS

# Run single test method
uv run pytest tests/test_geometry.py::TestNMS::test_no_overlap

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=bakery
```

## Project Phase Context

**Current Phase**: Luna 🌙 (Refactorization from monolithic to modular)

Architecture Evolution:
```
AS-IS (run_lens.py - 2,260 LOC) → Luna (Modular Package) → Juno (Pipeline) → Neon (Multi-Stream)
     Monolithic                        Current Phase
```

## Code Style Guidelines

### Project Structure
```
bakery/
├── core/entities/     # Domain entities (immutable dataclasses)
│   ├── frame.py       # Frame, CropInfo
│   ├── detection.py   # BoundingBox, Mask, Segmentation
│   ├── pose.py        # KeyPoint, Skeleton, PoseEstimation
│   └── model_config.py # ModelConfig, PipelineConfig, enums
├── adapters/          # External integrations (OpenVINO, Supervision, CLI)
├── utils/             # Shared utilities (geometry, metrics)
└── tests/             # Unit tests mirroring source structure
```

### Imports
- Standard library first, third-party second, local third
- Use absolute imports: `from bakery.core.entities import Frame`
- Group imports: stdlib, external (numpy, pytest), local
- Use `__all__` in `__init__.py` files to define public API

### Types & Naming
- Use type hints for all function signatures
- Use dataclasses for entities (`@dataclass(frozen=True)` for immutability)
- Classes: PascalCase (e.g., `BoundingBox`, `Segmentation`)
- Functions/variables: snake_case (e.g., `xywh2xyxy`, `crop_info`)
- Constants: UPPER_SNAKE_CASE (e.g., `COCO_KEYPOINT_NAMES`)
- Private methods: `_leading_underscore`

### Formatting
- Follow PEP 8
- 4 spaces indentation
- 88-100 character line length
- Double quotes for strings

### Validation & Error Handling
- Validate in `__post_init__` for dataclasses
- Raise `ValueError` for invalid inputs with descriptive messages
- Use specific exception types, avoid bare `except`
- Example: `raise ValueError(f"x2 must be >= x1, got x1={self.x1}, x2={self.x2}")`

### Documentation
- Module docstrings explaining purpose
- Class docstrings with Attributes section
- Function docstrings with Args, Returns, Example sections
- Use Google-style docstrings

### Patterns
- Immutable value objects: `@dataclass(frozen=True)` (e.g., `CropInfo`)
- Mutable entities: `@dataclass` with `.copy()` method (e.g., `Frame`)
- Properties for computed values (width, height, area, center)
- Class methods for alternative constructors (`from_array`, `from_xyxy`)
- Factory methods: `.empty()`, `.invalid()`
- Validation in `__post_init__` with descriptive error messages

### Testing
- Use pytest with class-based test organization
- Test class names: `Test{ClassName}` or `Test{FunctionName}`
- Descriptive test method names: `test_{scenario}_{expected}`
- Use `pytest.raises()` for exception testing
- Use `np.testing.assert_array_almost_equal()` for array comparisons
- Use `pytest.approx()` for floating point comparisons
- Mirror source structure in tests/

### Dependencies
- Core: numpy, openvino, supervision, ultralytics
- Dev: pytest, pytest-cov
- Python: >=3.11.14,<3.12

### uv Project Notes
- Use `uv run` prefix for all Python commands
- Dependencies managed in `pyproject.toml`
- Virtual environment at `.venv/`

## Key Architectural Principles

1. **Single Responsibility**: Each module handles one concern
2. **Immutability**: Value objects are frozen (`@dataclass(frozen=True)`)
3. **Validation**: Entities validate themselves in `__post_init__`
4. **Domain-Driven**: core/entities contain business logic
5. **Framework-Agnostic**: core/ doesn't depend on external libs (OpenVINO, etc.)
6. **Explicit Over Implicit**: All dependencies injected, no global state
