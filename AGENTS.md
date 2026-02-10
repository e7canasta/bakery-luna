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



Bakery - FP16 Segmentation + Pose Inference - SINGLE MODEL TRANSLUCENT WITH POSE
=================================================================================
Versión HYBRID: Segmentación para todos los objetos + Pose skeleton para personas.
Estilo visual Disney/Roger Rabbit con skeleton gris translúcido (uniforme y sutil).

Diferencias con versión original:
- Procesa 1 modelo segmentation + 1 modelo pose (hybrid approach)
- Segmentación: máscaras de color para TODOS los objetos
- Pose: skeleton (edges + vertices) solo para personas detectadas
- Estilo translúcido consistente en TODAS las anotaciones (mismo gris sutil)

Filosofía: "Complejidad por diseño, no por accidente"
- Inferencia optimizada OpenVINO FP16 en iGPU Intel Xe
- **Focus Lens**: Crop cuadrado configurable antes de inferir
- **Dual Model Pipeline**: Segmentation + Pose en paralelo
- **Anotaciones multicapa**: Background + Masks + Boxes + Labels + Skeleton
- Live preview opcional para debugging
- Métricas de performance medibles

Visualización estilo Disney/Roger Rabbit (puro + esquinas + skeleton):
  🎬 Mundo B&W: Frame en gris medio (oscurecido para contraste)
  🔍 Focus Lens Region: Sutilmente más claro que el frame (~10% más brillante)
  ✨ Halo: Glow suave alrededor de detecciones (define límites sin líneas)
  📐 Esquinas bbox: Gris claro translúcido (30% opacity) - marcan los límites del bbox
  🎨 Objetos detectados: COLOR COMPLETO + BRILLANTES (como toons iluminados)
  📊 Barra de confianza: Abajo DERECHA, gris translúcido (40% opacity, MUY discreta)
  🏷️ Labels: Abajo IZQUIERDA, TRANSLÚCIDO (40% opacity) - efecto fantasmal elegante
  🦴 Skeleton (Pose): Edges gris claro + Vertices gris medio (30% opacity) - sutil y uniforme
  
Estética clásica Disney (con esquinas + skeleton sutiles):
  - Frame base: gris medio oscurecido (0.6x → más contraste)
  - Focus lens: gris ligeramente más claro (1.1x del base)
  - Halo effect: Glow BLANCO/CELESTE (opacity=0.5) que define contornos como luz pura
  - Esquinas bbox: Gris claro translúcido (30% opacity) → marcan límites sin ser invasivas
  - Objetos detectados: Color original + brightness boost (1.2x) → "spotlight effect"
  - Labels & barra: Translúcidos (40% opacity) → efecto fantasmal
  - Skeleton pose: Gris claro/medio (30% opacity) → líneas y puntos sutiles, no invasivos
  - Efecto: objetos coloridos emergen con aura de luz + anotaciones grises fantasmales