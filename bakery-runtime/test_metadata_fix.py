import numpy as np
from bakery_runtime.processing.image import PreprocessCache

def test_preprocess_cache_metadata():
    cache = PreprocessCache()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_id = 0
    seg_shape = (320, 320)
    pose_shape = (320, 320)

    # 1. Identical shapes
    _, seg_meta, _, pose_meta = cache.get_or_compute(frame, frame_id, seg_shape, pose_shape)
    
    assert "input_shape" in seg_meta, "Failed: input_shape missing in seg_meta (identical shapes)"
    assert "input_shape" in pose_meta, "Failed: input_shape missing in pose_meta (identical shapes)"
    assert seg_meta["input_shape"] == seg_shape, f"Expected {seg_shape}, got {seg_meta.get('input_shape')}"

    # 2. Different shapes
    frame_id = 1
    pose_shape_diff = (480, 480)
    _, seg_meta_diff, _, pose_meta_diff = cache.get_or_compute(frame, frame_id, seg_shape, pose_shape_diff)
    
    assert "input_shape" in seg_meta_diff, "Failed: input_shape missing in seg_meta (different shapes)"
    assert "input_shape" in pose_meta_diff, "Failed: input_shape missing in pose_meta (different shapes)"
    assert seg_meta_diff["input_shape"] == seg_shape, f"Expected {seg_shape}, got {seg_meta_diff.get('input_shape')}"
    assert pose_meta_diff["input_shape"] == pose_shape_diff, f"Expected {pose_shape_diff}, got {pose_meta_diff.get('input_shape')}"

    print("Success: input_shape present in metadata")

if __name__ == "__main__":
    test_preprocess_cache_metadata()
