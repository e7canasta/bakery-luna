
import numpy as np
from bakery_runtime.processing.results import postprocess_segmentation

def test_class_filtering():
    print("🧪 Testing Class Filtering Logic...")

    # Mock parameters
    num_classes = 80
    mask_dim = 32
    num_anchors = 100
    conf_threshold = 0.25
    input_shape = (640, 640)

    # create a mock output tensor [1, 116, 100]
    # 4 box + 80 classes + 32 mask coefs
    output_boxes = np.zeros((1, 4 + num_classes + mask_dim, num_anchors), dtype=np.float32)
    
    # Create mock detections
    # Detection 0: Class 0 (Person), High Confidence
    output_boxes[0, 4, 0] = 0.9  # Class 0 score
    output_boxes[0, 0:4, 0] = [100, 100, 50, 50] # box

    # Detection 1: Class 1 (Bicycle), High Confidence
    output_boxes[0, 5, 1] = 0.9  # Class 1 score
    output_boxes[0, 0:4, 1] = [200, 200, 50, 50] # box
    
    # Detection 2: Class 0 (Person), Low Confidence
    output_boxes[0, 4, 2] = 0.1  # Class 0 score (below threshold)

    # Mock masks (random)
    output_masks = np.random.rand(1, mask_dim, 160, 160).astype(np.float32)

    # Case 1: No Filter
    print("\n[Case 1] No Filter")
    boxes, scores, class_ids, masks = postprocess_segmentation(
        output_boxes.copy(), output_masks, input_shape, conf_threshold
    )
    print(f"   Detections: {len(class_ids)}")
    print(f"   Classes: {class_ids}")
    assert len(class_ids) == 2, "Should detect 2 objects"
    assert 0 in class_ids and 1 in class_ids, "Should detect class 0 and 1"

    # Case 2: Filter Class 0
    print("\n[Case 2] Filter Class 0 (Person) Only")
    boxes, scores, class_ids, masks = postprocess_segmentation(
        output_boxes.copy(), output_masks, input_shape, conf_threshold, classes=[0]
    )
    print(f"   Detections: {len(class_ids)}")
    print(f"   Classes: {class_ids}")
    assert len(class_ids) == 1, "Should detect 1 object"
    assert class_ids[0] == 0, "Should only detect class 0"

    # Case 3: Filter Class 1
    print("\n[Case 3] Filter Class 1 (Bicycle) Only")
    boxes, scores, class_ids, masks = postprocess_segmentation(
        output_boxes.copy(), output_masks, input_shape, conf_threshold, classes=[1]
    )
    print(f"   Detections: {len(class_ids)}")
    print(f"   Classes: {class_ids}")
    assert len(class_ids) == 1, "Should detect 1 object"
    assert class_ids[0] == 1, "Should only detect class 1"

    print("\n✅ Verification Successful!")

if __name__ == "__main__":
    test_class_filtering()
