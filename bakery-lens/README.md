# bakery-lens

Crop-based inference optimization with adaptive strategies.

## Features

- **Static Focus Lens**: Fixed crop region for inference concentration
- **Adaptive Shift** (planned): Dynamic lateral crop adjustment based on detection feedback
- **Adaptive Expand** (planned): Dynamic crop size adjustment for multi-person scenarios

## Usage

```python
from bakery_lens import create_lens, FocusLensConfig

# Static lens (default)
config = FocusLensConfig(focus_size=432)
lens = create_lens(config)

# Process frame
result = lens.process(frame_data, frame_id=0)

# Map results back to full frame
detections = lens.map_detections(crop_detections)
keypoints = lens.map_keypoints(crop_keypoints)

# Feed back detections (no-op for static, used by adaptive)
lens.update(crop_detections)
```


# 1. Re-instalar dependencias (para linkear bakery-lens)
uv sync
# 2. Correr los tests migrados
uv run pytest bakery-lens/src/bakery_lens/tests/test_focus_lens.py
# 3. Correr tu comando original para asegurar integración
uv run bakery run --video var/data/videos/vador3.mp4 --models-dir models/ --show \
  --focus-size 512 --focus-strategy pad --seg-size m --pose-size x --resolution 320


Fase 1 Completa: Adaptive Shift Strategy

He implementado la estrategia de desplazamiento lateral adaptativo y la he integrado en el pipeline y la CLI.

Características:

Shift Dinámico: El crop se mueve lateralmente si detecta personas cerca de los bordes.
Suavizado: Uso de smoothing (Exponential Moving Average) para evitar movimientos bruscos.
Integración CLI: Nuevos argumentos disponibles en bakery run.
Tests Unitarios: Añadidos en bakery-lens/src/bakery_lens/tests/test_adaptive_shift.py.
Cómo probar: Puedes ejecutar el pipeline con el flag --adaptive y ver cómo el foco sigue a las personas (lateralmente):

bash
uv run bakery run --video var/data/videos/vador3.mp4 --models-dir models/ --show \
  --focus-size 432 --focus-strategy zoom \
  --adaptive --shift-step 40 --edge-threshold 80
(Nota: Ajusta shift-step y edge-threshold según tu video. Un threshold alto (80px) hará que reaccione antes).

Fase 2 Completa: Adaptive Expand Strategy

La implementación está terminada. He actualizado la estrategia AdaptiveShiftLens para soportar expansión dinámica cuando se habilita el flag --allow-expand.

Cambios Clave:

Lógica Expand: Si se detectan personas en ambos bordes simultáneamente, la ventana de crop aumenta de tamaño.
Decay: Si la "presión" desaparece, el tamaño vuelve gradualmente al original.
Protocolo: Actualicé LensStrategy para permitir tamaños variables (x, y, w, h).
CLI: Nuevo flag --allow-expand.
Verificación: He creado un walkthrough.md con los comandos para probar. Te sugiero correr:

Tests Unitarios (Verifican Shift y Expand):
bash
uv run pytest bakery-lens/src/bakery_lens/tests/test_adaptive_shift.py
Prueba en Video:
bash
uv run bakery run --video var/data/videos/vador3.mp4 --models-dir models/ --show --focus-size 432 --focus-strategy zoom --adaptive --allow-expand --shift-step 40 --edge-threshold 60