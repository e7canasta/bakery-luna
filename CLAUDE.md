# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest bakery/tests/ -v

# Run specific test file
uv run pytest bakery/tests/test_geometry.py -v

# Run specific test class
uv run pytest bakery/tests/test_geometry.py::TestNMS -v

# Run specific test method
uv run pytest bakery/tests/test_geometry.py::TestNMS::test_no_overlap -v

# Run with coverage
uv run pytest bakery/tests/ --cov=bakery --cov-report=html

# Run the main pipeline
uv run run_luna.py --video videos/sample.mp4 --models-dir exports/fp16/ --show
```

## Architecture Overview

Bakery is a dual-model vision pipeline (segmentation + pose estimation) optimized for OpenVINO runtime on Intel iGPU. The architecture follows Clean Architecture principles with dependencies pointing inward.

```
bakery/
├── core/entities/       # Domain entities (immutable dataclasses, no external deps)
│   ├── frame.py         # Frame, CropInfo
│   ├── detection.py     # BoundingBox, Mask, Segmentation
│   ├── pose.py          # KeyPoint, Skeleton, PoseEstimation
│   └── model_config.py  # ModelConfig, PipelineConfig, enums
├── adapters/openvino/   # External integrations
│   ├── model_repository.py  # Model discovery & validation
│   ├── inference_engine.py  # OpenVINO inference execution
│   ├── preprocessing.py     # Letterbox, normalization, PreprocessCache
│   └── postprocessing.py    # YOLO output parsing, NMS, mask generation
├── pipeline/            # Orchestration layer
│   └── dual_model_pipeline.py  # Dual-model coordinator with smart scheduling
├── annotators/          # Rendering
│   └── disney_annotator.py     # 8-layer Disney/Roger Rabbit aesthetic
└── utils/               # Shared utilities
    ├── geometry.py      # NMS, IoU, coordinate transforms
    └── metrics.py       # Performance tracking
```

### Key Architectural Patterns

1. **Smart Scheduling**: Segmentation runs every N frames (default 5), pose runs every frame. Cached segmentation is reused between seg runs.

2. **Preprocessing Cache**: When seg and pose models use the same resolution, tensors are computed once and reused (50% speedup).

3. **Entity-Driven Design**: Core entities are frozen dataclasses with validation in `__post_init__`. Use `.to_supervision()` to bridge to the supervision library ecosystem.

4. **Adapter Pattern**: OpenVINO, cv2, and supervision are isolated in adapters. Core logic has no external dependencies.

## Code Style

- Type hints for all function signatures
- `@dataclass(frozen=True)` for value objects
- Validation in `__post_init__` with descriptive ValueError messages
- Google-style docstrings with Args, Returns, Example sections
- Use absolute imports: `from bakery.core.entities import Frame`
- Use `__all__` in `__init__.py` files to define public API
- BDD-style tests (Given/When/Then in docstrings)

## Key Dependencies

- Python >=3.11.14,<3.12
- openvino==2024.0.0
- supervision>=0.26.1
- ultralytics>=8.3.222
- pytest (dev)

## Project Context

**Current Phase**: Luna (modular refactorization complete, 126 tests passing)
**Next Phase**: Juno (production CLI with video I/O threading, YAML config, metrics dashboard)

The pipeline processes video with a Disney/Roger Rabbit aesthetic: detected objects "pop" with color against a desaturated B&W background using 8-layer alpha-blended rendering.



akery - FP16 Segmentation + Pose Inference - SINGLE MODEL TRANSLUCENT WITH POSE
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