# ADRs y Specs para Bakery Pipeline

## Value Proposition

Bakery es un **edge-first stream video AI pipeline** diseñado como **composable building block**:

- **No es**: Web app, SaaS, fullstack framework
- **Sí es**: Pipeline de inferencia idiomático que puede ser el `on_video_frame` de sistemas como Roboflow InferencePipeline
- **Principio**: "No ahogar al rey" - no over-engineer threading/queues cuando otros lo resuelven

● El plan captura el value proposition que mencionaste:         

 Bakery como building block composable:   
 - process_frame() es el corazón - puede ser el on_video_frame de Roboflow
 - Roboflow maneja threading/buffering/reconnection (no ahogar al rey)        
 - .to_supervision() para interop con el ecosistema      
 
 ADRs propuestos:   
 1. 001 - OpenVINO: Por qué Intel iGPU edge, no NVIDIA 
 2. 002 - Dual-Model Pipeline: Smart scheduling, cache optimization 
 3. 003 - Disney Aesthetic: 8-layer rendering diferenciador                                                                                                                                                         
 4. 005 - Composable Design: Building block, no framework completo

● el value proposition real:         
 Bakery es un building block composable, no un framework completo.   
 Puede ser el on_video_frame de Roboflow InferencePipeline, delegando threading/buffering/reconnection. Esto es "no ahogar al rey" - no reinventar lo que Roboflow ya resuelve bien. 
 Déjame escribir el plan con los ADRs que capturan estas decisiones arquitectónicas clave.

## ADRs a Crear

| ADR | Título | Decisión Clave |
|-----|--------|----------------|
| 001 | OpenVINO as Inference Backend | Intel iGPU edge, no NVIDIA dependency |
| 002 | Dual-Model Pipeline Architecture | Seg + Pose separados, smart scheduling |
| 003 | Disney/Roger Rabbit Aesthetic | 8-layer rendering diferenciador |
| 005 | Composable Pipeline Design | Building block, no framework completo |

## Specs a Crear

| Spec | Descripción |
|------|-------------|
| dual-model-pipeline.md | API, scheduling, cache, metrics |
| disney-annotator.md | 8 layers, RenderConfig, extensibilidad |

---

## ADR-001: OpenVINO as Inference Backend

**Context**: Elegir runtime de inferencia para modelos YOLO
**Decision**: OpenVINO optimizado para Intel iGPU (Xe Graphics)
**Rationale**:
- Hardware consumer disponible (laptops Intel sin NVIDIA)
- FP16 eficiente en iGPU integrada
- Edge deployment sin dependencias CUDA
- Ultralytics exporta nativo a OpenVINO IR

**Alternatives Rejected**:
- ONNX Runtime: Peor performance en Intel iGPU
- TensorRT: Requiere NVIDIA GPU
- PyTorch: Sin optimización para edge

---

## ADR-002: Dual-Model Pipeline Architecture

**Context**: Procesar video con segmentación + pose estimation
**Decision**: Modelos separados con smart scheduling
**Rationale**:
- Seg es costoso (masks), pose es rápido (keypoints)
- seg_interval permite cachear segmentación entre frames
- Pose cada frame para skeleton suave
- PreprocessCache cuando resoluciones coinciden

**Pattern**:
```
Frame → PreprocessCache → [Seg (if N) | Pose (always)] → Entities → Annotator
```

---

## ADR-003: Disney/Roger Rabbit Aesthetic

**Context**: Diferenciar output visual
**Decision**: 8-layer alpha-blended rendering
**Rationale**:
- B&W world + color spotlight = objetos "pop"
- Único en el mercado (MediaPipe, Detectron2 no tienen estética)
- Configurable via RenderConfig
- Extensible (otros annotators posibles)

---

## ADR-005: Composable Pipeline Design

**Context**: Integración con ecosistema de stream processing
**Decision**: Diseño como building block, no framework completo
**Rationale**:
- **No ahogar al rey**: No reinventar threading/buffering
- Puede ser `on_video_frame` de Roboflow InferencePipeline
- DualModelPipeline.process_frame() es stateless (excepto cache)
- Entities con `.to_supervision()` para interop

**Integration Pattern**:
```python
from inference import InferencePipeline

def bakery_processor(frames: List[VideoFrame]) -> List[dict]:
    results = []
    for vf in frames:
        frame = Frame.from_array(vf.image, frame_id=vf.frame_id)
        seg, pose = pipeline.process_frame(frame)
        results.append({
            "detections": seg.to_supervision(),
            "keypoints": pose.to_supervision()
        })
    return results

# Roboflow maneja threads, buffers, reconnection
pipeline = InferencePipeline.init_with_custom_logic(
    video_reference="rtsp://stream",
    on_video_frame=bakery_processor,
    on_prediction=disney_sink
)
```

**Future Options** (no ahora):
- Propio VideoSource con threading (cuando sea necesario)
- Multi-stream support
- WebSocket output

---

## Files to Create

### docs/adrs/
- `001-openvino-as-inference-backend.md`
- `002-dual-model-pipeline-architecture.md`
- `003-disney-roger-rabbit-aesthetic.md`
- `005-composable-pipeline-design.md`

### docs/specs/
- `dual-model-pipeline.md`
- `disney-annotator.md`

---

## Verification

1. ADRs siguen template existente (`000-template.md`)
2. Specs siguen formato de `focus-lens.md`
3. Cross-references entre ADRs y specs
4. Value proposition claro en cada documento
