# bakery-runtime.filtering Protocol Specification

**Scope**: Composable post-inference filtering for detection and keypoint results.
**Target Audience**: AI Agents / Developers integrating filtering into inference pipelines.

## 1. Core Abstractions

### `FilterPolicy` (Value Object)
Immutable, declarative configuration for all filtering rules. No numpy dependency — pure data.
```python
@dataclass(frozen=True)
class FilterPolicy:
    # Detection
    global_confidence: float = 0.25
    seg_confidence: float | None = None
    pose_confidence: float | None = None
    classes: tuple[int, ...] | None = None
    per_class_confidence: dict[int, float] | None = None
    # Keypoints
    keypoint_confidence: float | None = None
    per_keypoint_confidence: dict[int, float] | None = None
    keypoint_min_visible: int | None = None
```

### Confidence Resolution Order
Most specific wins. Cascade for detections:
```
per_class_confidence[cls_id]  →  seg_confidence / pose_confidence  →  global_confidence
```
Cascade for keypoints:
```
per_keypoint_confidence[kp_id]  →  keypoint_confidence  →  global_confidence
```

### Resolver Methods
```python
policy.resolve_seg_confidence()       -> float
policy.resolve_pose_confidence()      -> float
policy.resolve_class_confidence(id)   -> float
policy.resolve_keypoint_confidence(id)-> float
```

### Predicates
```python
policy.has_class_filter         -> bool
policy.has_per_class_confidence -> bool
policy.has_keypoint_filter      -> bool
```

## 2. API Usage

### Construction
```python
from bakery_runtime.filtering import FilterPolicy

# Direct
policy = FilterPolicy(
    classes=(0, 2),
    per_class_confidence={0: 0.6, 2: 0.3},
    keypoint_min_visible=5,
)

# From CLI args (paired lists)
policy = FilterPolicy.from_cli(
    confidence=0.25,
    classes=[0, 2],
    class_confidence=[0.6, 0.3],   # zipped with classes
    keypoints=[9, 15],
    keypoint_confidence=[0.3, 0.4], # zipped with keypoints
)
```

### Filter Functions (Stateless, Numpy)
Execute **post-NMS, pre-entity construction** in the pipeline.

```python
from bakery_runtime.filtering import filter_detections, filter_keypoints

# Detections (seg output)
boxes, scores, class_ids, masks = filter_detections(
    boxes, scores, class_ids, masks, policy
)

# Keypoints (pose output)
keypoints, boxes, scores = filter_keypoints(
    keypoints, boxes, scores, policy
)
```

### `filter_detections` Behavior
1. **Class whitelist**: drop detections not in `policy.classes`
2. **Per-class confidence**: drop detections below `policy.resolve_class_confidence(cls_id)`

### `filter_keypoints` Behavior
1. **Per-keypoint threshold**: zero out `(x, y, conf)` if conf < resolved threshold
2. **Min visible**: drop entire skeleton if visible count < `policy.keypoint_min_visible`

## 3. COCO Keypoint Constants

```python
from bakery_runtime.filtering import COCO_KEYPOINTS, resolve_keypoint_id

COCO_KEYPOINTS["left_wrist"]   # → 9
resolve_keypoint_id("left_wrist")  # → 9
resolve_keypoint_id(9)             # → 9
resolve_keypoint_id("9")           # → 9
```

17 keypoints: `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`.

## 4. Integration Point

`DualModelPipeline` accepts `filter_policy: Optional[FilterPolicy]` in constructor. Filters are applied between `engine.postprocess()` and `_create_segmentation()` / `_create_pose_estimation()`.

## 5. Package Layout

```
bakery_runtime/filtering/
├── __init__.py      # Public exports
├── policy.py        # FilterPolicy dataclass
├── apply.py         # filter_detections, filter_keypoints
└── keypoints.py     # COCO constants + resolve_keypoint_id
```
