# bakery-lens Protocol Specification

**Scope**: High-performance dynamic cropping and coordinate mapping for inference pipelines.
**Target Audience**: AI Agents / Developers integrating focus capabilities.

## 1. Core Abstractions

### `Lens` Protocol
The primary interface for interaction.
```python
class Lens(Protocol):
    def process(self, frame: np.ndarray, frame_id: int) -> LensResult: ...
    def map_detections(self, detections: sv.Detections, crop_info: CropInfo, frame_w: int, frame_h: int) -> sv.Detections: ...
    def map_keypoints(self, keypoints: sv.KeyPoints, crop_info: CropInfo) -> sv.KeyPoints: ...
    def update(self, detections: sv.Detections) -> None: ...
```

### `LensStrategy` Protocol
Internal logic for determining crop parameters.
```python
class LensStrategy(Protocol):
    # Returns (x, y, w, h)
    def compute_crop_params(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]: ...
    def update(self, detections: sv.Detections) -> None: ...
```

### Data Types
- **`FocusLensConfig`**: Immutable configuration.
  - `focus_size`: Base size (int).
  - `adaptive`: Enable dynamic strategies (bool).
  - `allow_expand`: Enable size variation (bool).
  - `edge_threshold`: Pixel distance to trigger shift (int).
  - `shift_step`: Step size for movement (int).
  - `smoothing`: EMA factor [0.0-1.0].
- **`CropInfo`**: Metadata for a specific frame crop.
  - `x, y, width, height`: Crop region in original frame.
  - `scale_factor`: Scale applied (if zoom strategy).
  - `pad_x, pad_y`: Padding applied (if pad strategy with small frames).

## 2. API Usage

### Factory
Always use `create_lens` to instantiate.
```python
from bakery_lens import create_lens, FocusLensConfig

config = FocusLensConfig(focus_size=640, adaptive=True, allow_expand=True)
lens = create_lens(config)
```

### Processing Loop
The **Feedback Loop** is critical for adaptive behavior.
```python
# 1. Process
result = lens.process(frame, frame_id)
# result.frame is the crop (np.ndarray)
# result.crop_info is the metadata

# 2. Inference (External)
detections = model.predict(result.frame)

# 3. Feedback (CRITICAL)
lens.update(detections)

# 4. Global Mapping
global_detections = lens.map_detections(detections, result.crop_info, W, H)
```

## 3. Ops (Low Level)
Direct access to pure functions if bypassing the `Lens` object.
- `bakery_lens.ops.crop.apply_focus_lens`
- `bakery_lens.ops.mapping.map_detections_to_full_frame`

## 4. Strategies
- **Static**: Fixed position (`focus_x`, `focus_y`) or Centered. Fixed size.
- **AdaptiveShift**:
  - Tracks detections.
  - Shifts `x` if detections < `edge_threshold`.
  - Expands `w, h` if detections on BOTH edges (requires `allow_expand=True`).
  - Decays `w, h` if no edge pressure.
