# Bakery Vision Pipeline - Architecture & Specification

> **Version**: 1.0.0
> **Date**: 2026-02-09
> **Status**: Production Ready
> **Philosophy**: "Complejidad por diseno, no por accidente"

---

## Table of Contents

1. [C4 Model Architecture](#c4-model-architecture)
2. [Functional Specification](#functional-specification)
3. [Agent-Native API](#agent-native-api)
4. [Expert Analysis](#expert-analysis)

---

# C4 Model Architecture

## Level 1: System Context

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM CONTEXT                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌──────────────┐         ┌─────────────────────┐         ┌────────────┐  │
│    │              │         │                     │         │            │  │
│    │    User /    │────────▶│   Bakery Pipeline   │────────▶│   Output   │  │
│    │    Agent     │         │                     │         │   Video    │  │
│    │              │  Video  │  Vision Processing  │ Annotated│            │  │
│    └──────────────┘  Input  │     + Rendering     │  Frames  └────────────┘  │
│                             │                     │                          │
│                             └──────────┬──────────┘                          │
│                                        │                                     │
│                                        │ Uses                                │
│                                        ▼                                     │
│                    ┌───────────────────────────────────────┐                 │
│                    │         External Systems              │                 │
│                    ├───────────────────────────────────────┤                 │
│                    │  • OpenVINO Runtime (Intel iGPU)      │                 │
│                    │  • YOLO11 Models (Seg + Pose FP16)    │                 │
│                    │  • Supervision Library (Annotations)  │                 │
│                    │  • OpenCV (Video I/O)                 │                 │
│                    └───────────────────────────────────────┘                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Actors:**
- **User/Agent**: Human operator or LLM agent invoking pipeline
- **Video Input**: MP4, webcam stream, or image sequence
- **Output**: Annotated video with Disney/Roger Rabbit aesthetic

---

## Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            BAKERY CONTAINERS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         APPLICATION LAYER                              │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │ run_luna.py │  │run_origin.py│  │run_stack_*.py│ │  CLI Tools  │   │ │
│  │  │  (Full)     │  │ (Baseline)  │  │  (Testing)  │  │  (Future)   │   │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────────────┘   │ │
│  └─────────┼────────────────┼────────────────┼───────────────────────────┘ │
│            │                │                │                              │
│            ▼                ▼                ▼                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         PIPELINE LAYER                                 │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │                    DualModelPipeline                              │ │ │
│  │  │  • Smart Scheduling (seg every N frames, pose every frame)       │ │ │
│  │  │  • Preprocessing Cache (shared tensor when resolutions match)    │ │ │
│  │  │  • Entity Transformation (model coords → frame coords)           │ │ │
│  │  │  • Performance Metrics                                           │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                       │
│            ┌─────────────────────────┼─────────────────────────┐            │
│            ▼                         ▼                         ▼            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   CORE LAYER     │  │  ADAPTERS LAYER  │  │ ANNOTATORS LAYER │          │
│  │                  │  │                  │  │                  │          │
│  │  • Frame         │  │  • OpenVINO      │  │  • Disney        │          │
│  │  • Segmentation  │  │    - Inference   │  │    Annotator     │          │
│  │  • PoseEstimation│  │    - Repository  │  │    (8 layers)    │          │
│  │  • ModelConfig   │  │    - Pre/Post    │  │                  │          │
│  │  • PipelineConfig│  │  • Supervision   │  │                  │          │
│  │                  │  │                  │  │                  │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Level 3: Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMPONENT DETAIL                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        bakery/core/entities/                            ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    ││
│  │  │   Frame     │  │ Segmentation│  │PoseEstimation│ │ ModelConfig │    ││
│  │  │             │  │             │  │             │  │             │    ││
│  │  │ • data      │  │ • bboxes[]  │  │ • skeletons[]│ │ • model_path│    ││
│  │  │ • frame_id  │  │ • masks[]   │  │ • frame_id  │  │ • model_type│    ││
│  │  │ • width     │  │ • frame_id  │  │             │  │ • resolution│    ││
│  │  │ • height    │  │             │  │ Skeleton:   │  │ • device    │    ││
│  │  │ • crop_info │  │ to_super-   │  │ • keypoints │  │ • precision │    ││
│  │  │             │  │ vision()    │  │ • bbox      │  │ • confidence│    ││
│  │  │ from_array()│  │             │  │             │  │             │    ││
│  │  │ apply_crop()│  │ filter_by() │  │ to_super-   │  │ PipelineConf│    ││
│  │  │             │  │             │  │ vision()    │  │ • seg_interval│  ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                     bakery/adapters/openvino/                           ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         ││
│  │  │ ModelRepository │  │ InferenceEngine │  │  Preprocessing  │         ││
│  │  │                 │  │                 │  │                 │         ││
│  │  │ discover_models │  │ compile()       │  │ letterbox()     │         ││
│  │  │ get_model_by_   │  │ infer()         │  │ preprocess_     │         ││
│  │  │   type()        │  │ get_input_shape │  │   with_metadata │         ││
│  │  │ validate_model  │  │ get_device()    │  │ PreprocessCache │         ││
│  │  │ extract_metadata│  │                 │  │                 │         ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         ││
│  │                                                                         ││
│  │  ┌─────────────────────────────────────────────────────────────┐       ││
│  │  │                    Postprocessing                            │       ││
│  │  │  postprocess_segmentation() → boxes, scores, class_ids, masks│       ││
│  │  │  postprocess_pose()         → boxes, scores, class_ids, kpts │       ││
│  │  └─────────────────────────────────────────────────────────────┘       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                       bakery/pipeline/                                  ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │                    DualModelPipeline                             │   ││
│  │  │                                                                  │   ││
│  │  │  process_frame(Frame) → (Segmentation, PoseEstimation)          │   ││
│  │  │                                                                  │   ││
│  │  │  Internal Flow:                                                  │   ││
│  │  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │   ││
│  │  │  │Preprocess│──▶│ Infer    │──▶│Postproc  │──▶│ Entity   │      │   ││
│  │  │  │  Cache   │   │ (GPU)    │   │          │   │ Creation │      │   ││
│  │  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘      │   ││
│  │  │                                                                  │   ││
│  │  │  Optimizations:                                                  │   ││
│  │  │  • seg_interval: Run segmentation every N frames                │   ││
│  │  │  • cache reuse: Same tensor if seg_shape == pose_shape          │   ││
│  │  │  • metrics: Track total_frames, seg_runs, pose_runs             │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      bakery/annotators/                                 ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │                    DisneyAnnotator                               │   ││
│  │  │                                                                  │   ││
│  │  │  8-Layer Rendering Pipeline:                                     │   ││
│  │  │  ┌────────────────────────────────────────────────────────────┐ │   ││
│  │  │  │ L1: B&W World        │ Grayscale + darken (0.6x)           │ │   ││
│  │  │  │ L2: Focus Lens       │ Brighten ROI (1.1x)                 │ │   ││
│  │  │  │ L3: Halo Effect      │ Soft glow around detections         │ │   ││
│  │  │  │ L4: Box Corners      │ Subtle corner markers (30% opacity) │ │   ││
│  │  │  │ L5: Color Spotlight  │ Original color + boost (1.2x)       │ │   ││
│  │  │  │ L6: Confidence Bars  │ Percentage indicators (40% opacity) │ │   ││
│  │  │  │ L7: Labels           │ Class ID + confidence (40% opacity) │ │   ││
│  │  │  │ L8: Skeleton Overlay │ Pose edges + vertices (30% opacity) │ │   ││
│  │  │  └────────────────────────────────────────────────────────────┘ │   ││
│  │  │                                                                  │   ││
│  │  │  annotate(frame, detections, keypoints) → annotated_frame       │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Level 4: Code (Data Flow)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW SEQUENCE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐                                                                 │
│  │ cv2.read│                                                                 │
│  │  (BGR)  │                                                                 │
│  └────┬────┘                                                                 │
│       │ np.ndarray [H, W, 3]                                                 │
│       ▼                                                                      │
│  ┌─────────────────┐                                                         │
│  │ Frame.from_array│                                                         │
│  │   (frame_id)    │                                                         │
│  └────────┬────────┘                                                         │
│           │ Frame entity                                                     │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    DualModelPipeline.process_frame()                │    │
│  │  ┌──────────────────────────────────────────────────────────────┐  │    │
│  │  │ PreprocessCache.get_or_compute()                              │  │    │
│  │  │   frame.data → letterbox → normalize → NCHW → tensor         │  │    │
│  │  │   Returns: seg_tensor, seg_meta, pose_tensor, pose_meta       │  │    │
│  │  └──────────────────────────────────────────────────────────────┘  │    │
│  │                              │                                      │    │
│  │         ┌────────────────────┴────────────────────┐                │    │
│  │         ▼                                         ▼                │    │
│  │  ┌─────────────────┐                    ┌─────────────────┐        │    │
│  │  │ if frame_id %   │                    │ Pose Inference  │        │    │
│  │  │ seg_interval == 0│                   │ (every frame)   │        │    │
│  │  │ Seg Inference   │                    │                 │        │    │
│  │  └────────┬────────┘                    └────────┬────────┘        │    │
│  │           │                                      │                  │    │
│  │           ▼                                      ▼                  │    │
│  │  ┌─────────────────┐                    ┌─────────────────┐        │    │
│  │  │postprocess_seg()│                    │postprocess_pose()│       │    │
│  │  │→ boxes, masks   │                    │→ boxes, keypoints│       │    │
│  │  └────────┬────────┘                    └────────┬────────┘        │    │
│  │           │                                      │                  │    │
│  │           ▼                                      ▼                  │    │
│  │  ┌─────────────────┐                    ┌─────────────────┐        │    │
│  │  │_create_segmen-  │                    │_create_pose_    │        │    │
│  │  │tation()         │                    │estimation()     │        │    │
│  │  │• coord transform│                    │• coord transform│        │    │
│  │  │• mask resize    │                    │• keypoint scale │        │    │
│  │  └────────┬────────┘                    └────────┬────────┘        │    │
│  │           │                                      │                  │    │
│  │           ▼                                      ▼                  │    │
│  │  ┌─────────────────┐                    ┌─────────────────┐        │    │
│  │  │  Segmentation   │                    │ PoseEstimation  │        │    │
│  │  │    Entity       │                    │    Entity       │        │    │
│  │  └─────────────────┘                    └─────────────────┘        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│           │                                         │                        │
│           │ .to_supervision()                       │ .to_supervision()      │
│           ▼                                         ▼                        │
│  ┌─────────────────┐                       ┌─────────────────┐              │
│  │ sv.Detections   │                       │  sv.KeyPoints   │              │
│  │ • xyxy, mask    │                       │  • xy, confidence│             │
│  │ • confidence    │                       │  • class_id     │              │
│  │ • class_id      │                       │                 │              │
│  └────────┬────────┘                       └────────┬────────┘              │
│           │                                         │                        │
│           └──────────────────┬──────────────────────┘                        │
│                              ▼                                               │
│                    ┌─────────────────┐                                       │
│                    │DisneyAnnotator. │                                       │
│                    │annotate()       │                                       │
│                    │  8-layer render │                                       │
│                    └────────┬────────┘                                       │
│                             │                                                │
│                             ▼                                                │
│                    ┌─────────────────┐                                       │
│                    │ annotated_frame │                                       │
│                    │  [H, W, 3] BGR  │                                       │
│                    └────────┬────────┘                                       │
│                             │                                                │
│              ┌──────────────┴──────────────┐                                 │
│              ▼                             ▼                                 │
│     ┌─────────────────┐           ┌─────────────────┐                        │
│     │ cv2.VideoWriter │           │  cv2.imshow()   │                        │
│     │    .write()     │           │   (optional)    │                        │
│     └─────────────────┘           └─────────────────┘                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# Functional Specification

## 1. Core Capabilities

| Capability | Description | Status |
|------------|-------------|--------|
| **Instance Segmentation** | YOLO11 seg models with masks | ✅ Production |
| **Pose Estimation** | YOLO11 pose with 17 COCO keypoints | ✅ Production |
| **Dual-Model Pipeline** | Orchestrated seg + pose inference | ✅ Production |
| **Disney Aesthetic** | 8-layer Roger Rabbit style rendering | ✅ Production |
| **Smart Scheduling** | Segmentation every N frames | ✅ Production |
| **Preprocessing Cache** | Shared tensors for same resolution | ✅ Production |
| **Model Auto-Discovery** | Scan directory for valid models | ✅ Production |

## 2. Input/Output Specification

### Input Formats
```yaml
Video:
  - MP4, AVI, MOV (any OpenCV-compatible format)
  - Webcam stream (device index: 0, 1, ...)
  - Resolution: any (letterbox preserves aspect ratio)

Models:
  - Format: OpenVINO IR (.xml + .bin)
  - Precision: FP16 recommended for iGPU
  - Naming: *-seg*.xml (segmentation), *-pose*.xml (pose)
  - Location: single directory, recursive discovery
```

### Output Formats
```yaml
Video:
  - Format: MP4 (mp4v codec)
  - Resolution: same as input
  - Annotations: Disney/Roger Rabbit aesthetic

Metrics:
  - total_frames: int
  - seg_runs: int
  - pose_runs: int
  - fps: float (computed)
```

## 3. Configuration Parameters

### PipelineConfig
```python
@dataclass
class PipelineConfig:
    segmentation: ModelConfig      # Segmentation model config
    pose: ModelConfig              # Pose estimation model config
    seg_interval: int = 5          # Run seg every N frames (1-30)
    confidence_threshold: float = 0.25  # Detection threshold (0.0-1.0)
    class_filter: List[int] = None     # COCO class IDs to keep
```

### RenderConfig (DisneyAnnotator)
```python
@dataclass
class RenderConfig:
    # Layer 1: B&W World
    bw_darkness: float = 0.6       # Darken factor (0.0-1.0)

    # Layer 2: Focus Lens
    lens_brightness: float = 1.1   # Brighten factor (1.0-2.0)

    # Layer 3: Halo Effect
    halo_color: Tuple = (240, 248, 255)  # RGB
    halo_opacity: float = 0.5      # 0.0-1.0
    halo_kernel_size: int = 40     # Blur size

    # Layer 4: Box Corners
    corner_color: Tuple = (200, 200, 200)
    corner_thickness: int = 1
    corner_length: int = 15
    corner_opacity: float = 0.3

    # Layer 5: Color Spotlight
    spotlight_brightness: float = 1.2

    # Layer 6-7: Labels & Bars
    bar_opacity: float = 0.4
    label_opacity: float = 0.4

    # Layer 8: Skeleton
    skeleton_opacity: float = 0.3
```

## 4. CLI Usage

```bash
# Full pipeline with visualization
uv run run_luna.py \
  --video videos/sample.mp4 \
  --models-dir exports/fp16/ \
  --output results/output.mp4 \
  --seg-interval 5 \
  --confidence 0.25 \
  --classes 0 \        # Optional: filter to persons only
  --show               # Optional: live preview

# Webcam mode
uv run run_luna.py --video 0 --models-dir exports/fp16/ --show
```

---

# Agent-Native API

## Design Philosophy

Bakery is designed for **agent-native operation**:

1. **Stateless Processing**: Each `process_frame()` call is independent
2. **Structured Outputs**: Domain entities with `.to_supervision()` for interop
3. **Fail-Fast Validation**: Errors at construction, not runtime
4. **Observable Metrics**: `get_metrics()` returns structured data
5. **Deterministic**: Same input → same output (no hidden state)

## Programmatic API

```python
from pathlib import Path
from bakery.core.entities import Frame, PipelineConfig
from bakery.adapters.openvino import ModelRepository, InferenceEngine
from bakery.pipeline import DualModelPipeline
from bakery.annotators import DisneyAnnotator, RenderConfig
from bakery.core.entities.model_config import ModelType

# 1. Discover models
repo = ModelRepository(Path("exports/fp16/"))
seg_model = repo.get_model_by_type(ModelType.SEGMENTATION)
pose_model = repo.get_model_by_type(ModelType.POSE)

# 2. Create engines
seg_engine = InferenceEngine(seg_model)
pose_engine = InferenceEngine(pose_model)

# 3. Configure pipeline
config = PipelineConfig(
    segmentation=seg_model,
    pose=pose_model,
    seg_interval=5,
    confidence_threshold=0.25
)

# 4. Create pipeline
pipeline = DualModelPipeline(seg_engine, pose_engine, config)

# 5. Process frames
frame = Frame.from_array(image_bgr, frame_id=0)
segmentation, pose_estimation = pipeline.process_frame(frame)

# 6. Convert to supervision (for annotation or further processing)
detections = segmentation.to_supervision()  # sv.Detections
keypoints = pose_estimation.to_supervision()  # sv.KeyPoints

# 7. Annotate
annotator = DisneyAnnotator(RenderConfig(), enable_pose=True)
result = annotator.annotate(frame.data, detections, keypoints)

# 8. Get metrics
metrics = pipeline.get_metrics()
print(f"Processed {metrics.total_frames} frames")
print(f"Seg efficiency: {metrics.total_frames / metrics.seg_runs:.1f}x")
```

## Agent Integration Patterns

### Pattern 1: Stream Processing
```python
# For real-time processing with agents
def process_video_stream(video_path: str, callback):
    """Agent-friendly stream processor with callback."""
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_entity = Frame.from_array(frame, frame_id=cap.get(cv2.CAP_PROP_POS_FRAMES))
        seg, pose = pipeline.process_frame(frame_entity)

        # Callback with structured data for agent consumption
        callback({
            "frame_id": frame_entity.frame_id,
            "detections": len(seg),
            "persons": len(pose),
            "keypoints": [s.to_array() for s in pose.skeletons]
        })
```

### Pattern 2: Batch Analysis
```python
# For offline analysis by agents
def analyze_video(video_path: str) -> dict:
    """Return structured analysis for agent consumption."""
    results = {
        "frames_analyzed": 0,
        "total_detections": 0,
        "persons_detected": 0,
        "avg_confidence": 0.0,
        "timeline": []
    }

    # ... process video ...

    return results  # JSON-serializable for agent parsing
```

---

# Expert Analysis

## Comparison with Industry Solutions

### vs. MediaPipe (Google)
| Aspect | Bakery | MediaPipe |
|--------|--------|-----------|
| **Architecture** | Modular, entity-driven | Monolithic graph-based |
| **Hardware** | Intel iGPU optimized | Multi-platform |
| **Customization** | Full control | Limited config |
| **Aesthetic** | Disney/Roger Rabbit | None (raw output) |
| **Model Format** | OpenVINO IR | TFLite/custom |

**Bakery Advantage**: Complete control over rendering aesthetic and pipeline optimization for specific hardware.

### vs. Detectron2 (Meta)
| Aspect | Bakery | Detectron2 |
|--------|--------|------------|
| **Complexity** | ~2000 LoC | ~100K LoC |
| **Learning Curve** | Hours | Weeks |
| **Deployment** | Single dir | Complex deps |
| **Focus** | Inference + Aesthetic | Training + Research |

**Bakery Advantage**: Purpose-built for inference with aesthetic rendering, not research.

### vs. Ultralytics YOLO
| Aspect | Bakery | Ultralytics |
|--------|--------|-------------|
| **Runtime** | OpenVINO (iGPU) | PyTorch/ONNX |
| **Entity Model** | Rich domain entities | Raw tensors |
| **Rendering** | 8-layer Disney | Basic boxes |
| **Dual-Model** | Native support | Manual orchestration |

**Bakery Advantage**: Domain-driven design with sophisticated rendering pipeline.

## Architectural Strengths

### 1. Clean Architecture Adherence
```
┌─────────────────────────────────────────────┐
│              Frameworks & Drivers           │  ← OpenCV, OpenVINO
├─────────────────────────────────────────────┤
│                 Adapters                    │  ← InferenceEngine, Repository
├─────────────────────────────────────────────┤
│               Use Cases                     │  ← DualModelPipeline
├─────────────────────────────────────────────┤
│                Entities                     │  ← Frame, Segmentation, Pose
└─────────────────────────────────────────────┘
```

Dependencies point **inward**. Core entities have zero external dependencies.

### 2. Entity-Driven Design
- **Immutable by default**: `@dataclass(frozen=True)` where appropriate
- **Self-validating**: `__post_init__` catches errors at construction
- **Interoperable**: `.to_supervision()` bridges to ecosystem

### 3. Optimization Strategy
- **Preprocessing cache**: O(1) memory, avoids redundant computation
- **Smart scheduling**: Segmentation is expensive, reuse between frames
- **Letterbox preservation**: Maintains aspect ratio, prevents distortion

## Recommendations for Evolution

### Phase 1: Stability (Current)
- [x] Core entities with validation
- [x] Dual-model pipeline with caching
- [x] Disney annotator with 8 layers
- [x] Model auto-discovery

### Phase 2: Enhancement
- [ ] Focus Lens integration (crop-based inference)
- [ ] Tracking integration (ByteTrack/BoT-SORT)
- [ ] Async inference (parallel seg + pose)
- [ ] WebSocket streaming output

### Phase 3: Scale
- [ ] Multi-camera support
- [ ] Distributed processing
- [ ] Cloud deployment (containerized)
- [ ] REST/gRPC API

## Final Assessment

**Bakery represents a well-designed, purpose-built vision pipeline** that achieves:

1. **Simplicity**: ~2000 LoC for complete functionality
2. **Performance**: Optimized for Intel iGPU with FP16
3. **Aesthetics**: Unique Disney/Roger Rabbit rendering
4. **Maintainability**: Clean architecture, domain entities
5. **Agent-Native**: Structured outputs, observable state

The design philosophy of "complejidad por diseno, no por accidente" is evident throughout. The codebase avoids over-engineering while maintaining clear separation of concerns.

**Comparison to similar solutions**: Bakery occupies a unique niche—it's simpler than research frameworks (Detectron2), more customizable than consumer APIs (MediaPipe), and more aesthetically sophisticated than raw inference tools (Ultralytics). The focus on Intel iGPU optimization makes it particularly suitable for edge deployment scenarios where NVIDIA GPUs are unavailable.

---
 Lo que Bakery hace bien        
  
 1. Simplicidad funcional con sofisticacion visual            
 - ~2000 LoC para pipeline completo (vs ~100K en Detectron2)            
 - 8 capas de rendering que logran una estetica unica         
 - El efecto Disney/Roger Rabbit es diferenciador real en el mercado    
  
 2. Arquitectura limpia sin over-engineering        
 - Entidades inmutables con validacion en construccion        
 - Adaptadores desacoplados (OpenVINO puede ser reemplazado)            
 - .to_supervision() como puente al ecosistema      
  
 3. Optimizaciones donde importan         
 - seg_interval: la segmentacion es costosa, reutilizarla es inteligente          
 - PreprocessCache: evita recomputar cuando las resoluciones coinciden            
 - FP16 en iGPU Intel: aprovecha hardware consumer            
  
 Comparacion con el sector      
 ┌─────────────┬─────────────┬─────────────┬─────────────────┐          
 │  Solucion   │ Complejidad │  Estetica   │ Hardware Target │          
 ├─────────────┼─────────────┼─────────────┼─────────────────┤          
 │ MediaPipe   │ Media       │ Ninguna     │ Mobile/Multi    │          
 ├─────────────┼─────────────┼─────────────┼─────────────────┤          
 │ Detectron2  │ Alta        │ Ninguna     │ NVIDIA GPU      │          
 ├─────────────┼─────────────┼─────────────┼─────────────────┤          
 │ Ultralytics │ Baja        │ Basica      │ Multi │          
 ├─────────────┼─────────────┼─────────────┼─────────────────┤          
 │ Bakery      │ Baja        │ Sofisticada │ Intel iGPU      │          
 └─────────────┴─────────────┴─────────────┴─────────────────┘          
 Bakery ocupa un nicho unico: pipelines de vision con estetica cinematografica en hardware consumer.            
  
 Recomendaciones      
  
 1. Focus Lens deberia integrarse - ya tienes la logica en run_lens.origin.py     
 2. Tracking (ByteTrack) seria el siguiente paso natural para coherencia temporal           
 3. API REST/WebSocket para integracion con sistemas externos           
  
 La filosofia "complejidad por diseno" se refleja bien. El codigo es legible, las responsabilidades estan claras, y las optimizaciones estan donde generan impacto real.    
  
 Veredicto: Bakery es un ejemplo de como disenar software de vision que balancea funcionalidad, rendimiento y estetica sin caer en la trampa del over-engineering.

*Generated by Claude Opus 4.5 - Architecture Analysis*
