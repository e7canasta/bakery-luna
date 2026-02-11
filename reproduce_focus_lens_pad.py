
import numpy as np
import supervision as sv
from bakery.core.entities.frame import Frame
from bakery.core.entities.focus_lens_config import FocusLensConfig
from bakery.utils.focus_lens import apply_focus_lens, map_detections_to_full_frame

def reproduce_issue():
    # Setup: 768x432 frame (as in user report)
    # Target: 512x512 crop
    # Strategy: pad
    
    # Create dummy frame
    frame_w, frame_h = 768, 432
    frame_data = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    frame = Frame.from_array(frame_data, frame_id=1)
    
    # Configure Focus Lens
    config = FocusLensConfig(focus_size=512, strategy="pad")
    
    # Apply Focus Lens
    cropped_frame, crop_info = apply_focus_lens(frame, config)
    
    print(f"Original Frame: {frame_w}x{frame_h}")
    print(f"Focus Size: {config.focus_size}")
    print(f"Crop Info: x={crop_info.x}, y={crop_info.y}, w={crop_info.width}, h={crop_info.height}, scale={crop_info.scale_factor}")
    
    # Calculation of expected padding
    # frame_h = 432, focus_size = 512
    # pad_h = 512 - 432 = 80
    # pad_top = 40, pad_bottom = 40
    expected_pad_top = 40
    
    # Simulate a detection at a specific point in the *original* frame content
    # Let's say there is an object at (100, 10) in the original frame.
    # In the padded frame, this pixel is shifted down by pad_top.
    # Padded Y = 10 + 40 = 50.
    # Crop Y start is likely 0 (since 512-512 // 2 = 0).
    # So in the cropped frame, the object is at (100, 50).
    
    # The model sees the cropped frame and detects the object at (100, 50).
    simulated_detection_xyxy = np.array([[100, 50, 120, 70]])  # x1, y1, x2, y2
    detections = sv.Detections(
        xyxy=simulated_detection_xyxy,
        confidence=np.array([0.9]),
        class_id=np.array([0])
    )
    
    # Map back to full frame
    mapped_detections = map_detections_to_full_frame(
        detections, crop_info, frame_w, frame_h
    )
    
    mapped_y1 = mapped_detections.xyxy[0][1]
    
    print(f"Simulated Detection (Crop): y1={simulated_detection_xyxy[0][1]}")
    print(f"Mapped Detection (Full): y1={mapped_y1}")
    print(f"Expected Detection (Original): y1={10}")
    
    # Check if mismatch equals padding
    diff = mapped_y1 - 10
    print(f"Difference: {diff}")
    
    if diff == expected_pad_top:
        print("FAIL: Mapped coordinate includes padding! Issue reproduced.")
    elif diff == 0:
        print("PASS: Mapped coordinate is correct.")
    else:
        print(f"FAIL: Unexpected difference {diff}")

if __name__ == "__main__":
    reproduce_issue()
