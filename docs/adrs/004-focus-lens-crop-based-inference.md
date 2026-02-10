# ADR-004: Focus Lens - Crop-Based Inference

**Status:** Accepted
**Date:** 2026-02-10

Decisión arquitectónica: por qué crop-based inference, alternativas consideradas, trade-offs

## Context


En pipelines de visión por computadora, procesar frames completos de alta resolución (1920x1080, 4K) tiene trade-offs significativos:

1. **Resolución de inferencia fija**: Los modelos YOLO operan en resoluciones cuadradas (320, 640px). Un frame 1920x1080 se reduce a 640x360 efectivos después de letterbox, perdiendo detalle.

2. **Objetos pequeños**: En frames de vigilancia/deportes, las personas ocupan una fracción pequeña del frame. La resolución efectiva por objeto es baja.

3. **Costo computacional**: Procesar el frame completo cuando solo interesa una región es ineficiente.

**Filosofía "Batalla Naval"**: Concentrar píxeles en la región de interés, como enfocar un lente en una zona específica.

## Decision

Implementar **Focus Lens**: un sistema de crop cuadrado configurable que se aplica ANTES de la inferencia.

### Arquitectura

```
Frame Completo [1920x1080]
        │
        ▼
    Focus Lens (crop 640x640 centrado)
        │
        ▼
Frame Cropped [640x640]
        │
        ▼
    Letterbox → Modelo (640x640)
        │
        ▼
    Detecciones (en coords del crop)
        │
        ▼
    Mapeo inverso (offset + scale)
        │
        ▼
Detecciones en coords del frame completo
```

### Componentes

1. **FocusLensConfig** (entidad inmutable):
   - `focus_size`: Tamaño del crop (múltiplo de 80)
   - `focus_x`, `focus_y`: Posición (None = centrado)
   - `strategy`: "zoom" | "pad" (para frames pequeños)

2. **Utilidades** (`bakery/utils/focus_lens.py`):
   - `apply_focus_lens()`: Aplica crop al frame
   - `map_detections_to_full_frame()`: Mapea bboxes y masks
   - `map_keypoints_to_full_frame()`: Mapea keypoints

3. **Integración en Pipeline**:
   - `DualModelPipeline` acepta `focus_lens_config` opcional
   - Aplica Focus Lens antes de preprocessing
   - Mapea resultados automáticamente a coords del frame completo

## Consequences

### Positive

- **Mayor densidad de píxeles por objeto**: Un objeto que ocupaba 50x50px en frame completo ahora ocupa ~150x150px en el crop
- **Mejor detección de objetos pequeños**: Más detalle para el modelo
- **Eficiencia**: Solo se procesa la región de interés
- **Backward compatible**: Focus Lens es opcional, pipeline funciona igual sin él
- **Integración con anotador**: DisneyAnnotator ya soporta visualizar la región de focus

### Negative

- **Campo de visión reducido**: Objetos fuera del crop no se detectan
- **Requiere conocimiento a priori**: Hay que saber dónde enfocar (o usar tracking para seguir objetos)
- **Complejidad adicional**: Transformación de coordenadas entre espacios

### Neutral

- El scale_factor de zoom se propaga para transformaciones inversas correctas
- Keypoints invisibles (0,0) no se transforman (convención mantenida)

## Alternatives Considered

### Alternative 1: Tiling (dividir frame en tiles)

Dividir el frame en múltiples tiles, inferir cada uno, y fusionar resultados.

**Rechazado porque:**
- Complejidad de fusión (NMS entre tiles)
- Latencia multiplicada por número de tiles
- No hay control sobre región de interés específica

### Alternative 2: Multi-scale inference

Procesar el frame a múltiples escalas y fusionar detecciones.

**Rechazado porque:**
- Costo computacional significativo
- Complejidad de fusión multi-escala
- No resuelve el problema de focalizar en región específica

### Alternative 3: Crop post-inferencia

Hacer inferencia en frame completo y filtrar detecciones por región.

**Rechazado porque:**
- No mejora la resolución efectiva por objeto
- Mismo costo computacional que sin focus
- No aprovecha la ventaja de concentrar píxeles

## References

- [YOLO Letterbox Preprocessing](https://docs.ultralytics.com/modes/predict/)
- [OpenVINO Model Optimization](https://docs.openvino.ai/)
- `bakery/utils/focus_lens.py` - Implementación
- `bakery/core/entities/focus_lens_config.py` - Entidad de configuración
