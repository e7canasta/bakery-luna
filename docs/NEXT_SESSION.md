# Next Session Context

> Contexto para continuar trabajo en Bakery Pipeline

## Value Proposition (Crítico)

Bakery es un **edge-first stream video AI pipeline** diseñado como **composable building block**:

- **No es**: Web app, SaaS, fullstack framework
- **Sí es**: Pipeline de inferencia que puede ser el `on_video_frame` de Roboflow InferencePipeline
- **Principio**: "No ahogar al rey" - no reinventar threading/queues cuando otros lo resuelven

## Decisiones Arquitectónicas Clave

1. **OpenVINO Backend**: Intel iGPU edge, no NVIDIA dependency
2. **Dual-Model Pipeline**: Seg + Pose separados, smart scheduling (seg_interval)
3. **Disney Aesthetic**: 8-layer rendering diferenciador
4. **Composable Design**: Building block que trabaja con Roboflow InferencePipeline

## Integración con Roboflow InferencePipeline

```python
from inference import InferencePipeline

def bakery_processor(frames: List[VideoFrame]) -> List[dict]:
    """Bakery como on_video_frame de Roboflow."""
    results = []
    for vf in frames:
        frame = Frame.from_array(vf.image, frame_id=vf.frame_id)
        seg, pose = pipeline.process_frame(frame)
        results.append({
            "detections": seg.to_supervision(),
            "keypoints": pose.to_supervision()
        })
    return results

# Roboflow maneja: threads, buffers, reconnection, sinks
pipeline = InferencePipeline.init_with_custom_logic(
    video_reference="rtsp://stream",
    on_video_frame=bakery_processor,
    on_prediction=disney_sink
)
```

## Trabajo Pendiente

### ADRs a Crear
| ADR | Título |
|-----|--------|
| 001 | OpenVINO as Inference Backend |
| 002 | Dual-Model Pipeline Architecture |
| 003 | Disney/Roger Rabbit Aesthetic |
| 005 | Composable Pipeline Design |

### Specs a Crear
- `dual-model-pipeline.md`
- `disney-annotator.md`

## Estado Actual

- ✅ Focus Lens implementado y testeado
- ✅ ADR-004 (Focus Lens) creado
- ✅ Spec Focus Lens creado
- ✅ Estructura docs/adrs/ y docs/specs/ lista
- 🔲 Faltan 4 ADRs y 2 Specs

## Referencias Importantes

- `references/roboflow-inference/4-stream-processing.md` - Arquitectura de InferencePipeline
- `references/roboflow-inference/4.1-inferencepipeline.md` - Threading model, sinks
- `docs/ARCHITECTURE.md` - Arquitectura actual de Bakery
- `docs/C4_MODEL.md` - Diagramas C4

## Filosofía de Diseño

> "Complejidad por diseño, no por accidente"

- Simplicidad funcional con sofisticación visual
- Arquitectura limpia sin over-engineering
- Optimizaciones donde importan (cache, scheduling)
- Entity-driven con `.to_supervision()` para interop
