# Feature Spec: Focus Lens

**Version:** 1.0.0
**Status:** Implemented

API completa, ejemplos de uso, CLI, coordinate transformation, testing

## Overview

Focus Lens es un sistema de crop cuadrado configurable que concentra la inferencia en una región de interés del frame, mejorando la densidad de píxeles por objeto detectado.

## API

### FocusLensConfig

```python
from bakery.core.entities import FocusLensConfig

# Configuración básica (centrado)
config = FocusLensConfig(focus_size=640)

# Con posición específica
config = FocusLensConfig(
    focus_size=480,
    focus_x=100,
    focus_y=200
)

# Con estrategia para frames pequeños
config = FocusLensConfig(
    focus_size=640,
    strategy="pad"  # o "zoom" (default)
)
```

#### Parámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `focus_size` | int | requerido | Tamaño del crop cuadrado. **Debe ser múltiplo de 80**. |
| `focus_x` | int \| None | None | Posición X del crop. None = centrado. |
| `focus_y` | int \| None | None | Posición Y del crop. None = centrado. |
| `strategy` | str | "zoom" | Estrategia cuando frame < focus_size: "zoom" o "pad". |

#### Validaciones

- `focus_size` debe ser múltiplo de 80 (80, 160, 240, 320, 400, 480, 560, 640...)
- `focus_size` debe ser positivo
- `focus_x`, `focus_y` deben ser no-negativos si se especifican
- `strategy` debe ser "zoom" o "pad"

### Utilities

```python
from bakery.utils import (
    apply_focus_lens,
    map_detections_to_full_frame,
    map_keypoints_to_full_frame,
    crop_info_to_tuple
)
```

#### apply_focus_lens

```python
def apply_focus_lens(
    frame: Frame,
    config: FocusLensConfig
) -> Tuple[Frame, CropInfo]:
    """
    Aplica focus lens a un frame.

    Returns:
        (cropped_frame, crop_info)
    """
```

#### map_detections_to_full_frame

```python
def map_detections_to_full_frame(
    detections: sv.Detections,
    crop_info: CropInfo,
    full_width: int,
    full_height: int
) -> sv.Detections:
    """
    Mapea detecciones del crop al frame completo.
    Incluye bboxes y masks.
    """
```

#### map_keypoints_to_full_frame

```python
def map_keypoints_to_full_frame(
    keypoints: sv.KeyPoints,
    crop_info: CropInfo
) -> sv.KeyPoints:
    """
    Mapea keypoints del crop al frame completo.
    Keypoints invisibles (0,0) no se transforman.
    """
```

### Pipeline Integration

```python
from bakery.core.entities import FocusLensConfig
from bakery.pipeline import DualModelPipeline

# Crear pipeline con Focus Lens
focus_config = FocusLensConfig(focus_size=640)
pipeline = DualModelPipeline(
    seg_engine,
    pose_engine,
    pipeline_config,
    focus_lens_config=focus_config  # Opcional
)

# Procesar frame (Focus Lens se aplica automáticamente)
segmentation, pose = pipeline.process_frame(frame)
# Resultados ya están en coordenadas del frame completo

# Obtener crop info para anotación
crop_info = pipeline.get_crop_info()
```

## CLI Usage

```bash


# Focus Lens centrado 640x640
uv run run_luna.py --video sample.mp4 --models-dir exports/fp16/ --focus-size 640

# Diferentes configuraciones
uv run run_luna.py --video videos/sample.mp4 --models-dir exports/fp16/ --focus-size 480 --focus-x 100 --focus-y 100 --show

# Focus Lens en posición específica
uv run run_luna.py --video sample.mp4 --models-dir exports/fp16/ \
    --focus-size 480 --focus-x 100 --focus-y 200

# Con estrategia pad
uv run run_luna.py --video sample.mp4 --models-dir exports/fp16/ \
    --focus-size 640 --focus-strategy pad
```

### CLI Arguments

| Argumento | Descripción |
|-----------|-------------|
| `--focus-size` | Tamaño del crop (múltiplo de 80) |
| `--focus-x` | Posición X (default: centrado) |
| `--focus-y` | Posición Y (default: centrado) |
| `--focus-strategy` | "zoom" o "pad" (default: zoom) |

## Strategies

### Zoom Strategy (default)

Cuando el frame es más pequeño que `focus_size`, se escala hacia arriba:

```
Frame 400x400, focus_size=640
→ Scale up a 640x640 (scale_factor=1.6)
→ Crop centrado 640x640
→ Coordenadas se dividen por scale_factor al mapear de vuelta
```

### Pad Strategy

Cuando el frame es más pequeño que `focus_size`, se rellena con negro:

```
Frame 400x400, focus_size=640
→ Pad a 640x640 con bordes negros
→ Crop (que incluye el padding)
→ scale_factor=1.0 (sin escalado)
```

## Coordinate Transformation

### Del Crop al Frame Completo

```python
# Sin zoom (scale_factor=1.0)
full_x = crop_x + crop_info.x
full_y = crop_y + crop_info.y

# Con zoom (scale_factor > 1.0)
full_x = (crop_x / scale_factor) + (crop_info.x / scale_factor)
full_y = (crop_y / scale_factor) + (crop_info.y / scale_factor)
```

### Masks

Las máscaras se expanden al tamaño del frame completo:
1. Si hubo zoom, se redimensiona la máscara al tamaño original
2. Se crea máscara vacía del tamaño del frame completo
3. Se coloca la máscara del crop en la posición correcta

### Keypoints

- Keypoints visibles se transforman con offset + inverse scale
- Keypoints invisibles (0,0) permanecen en (0,0) - convención estándar

## Visual Feedback

El `DisneyAnnotator` muestra la región del Focus Lens:

```python
annotator.annotate(
    frame=full_frame,
    detections=detections,
    keypoints=keypoints,
    focus_region=(crop_x, crop_y, width, height)  # Opcional
)
```

- Región del focus lens ligeramente más brillante
- Borde sutil gris alrededor de la región
- Detecciones renderizadas en coordenadas del frame completo

## Performance Considerations

| Escenario | Beneficio |
|-----------|-----------|
| Frame 1920x1080, focus 640x640 | 9x más píxeles por objeto en la región |
| Frame 4K, focus 640x640 | 36x más píxeles por objeto en la región |
| Objetos pequeños (50x50px) | Mejor detección por mayor resolución efectiva |

## Testing

```bash
# Tests unitarios de Focus Lens
uv run pytest bakery/tests/test_focus_lens.py -v

# Tests específicos
uv run pytest bakery/tests/test_focus_lens.py::TestFocusLensConfig -v
uv run pytest bakery/tests/test_focus_lens.py::TestApplyFocusLens -v
uv run pytest bakery/tests/test_focus_lens.py::TestMapDetectionsToFullFrame -v
```

## Related

- **ADR:** `docs/adrs/004-focus-lens-crop-based-inference.md`
- **Implementation:** `bakery/utils/focus_lens.py`
- **Entity:** `bakery/core/entities/focus_lens_config.py`
- **Tests:** `bakery/tests/test_focus_lens.py`
