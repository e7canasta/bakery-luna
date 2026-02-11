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
