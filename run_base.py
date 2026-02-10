"""
Bakery - run_origin.py - Luna Components with Original Flow
============================================================

This script uses Luna's modular components but follows the original
run_lens.origin.py flow for validation. It bypasses the entity layer
to test that core Luna components work correctly.

Usage:
    uv run run_origin.py --video videos/sample.mp4 --models-dir exports/fp16/
    uv run run_origin.py --video videos/sample.mp4 --models-dir exports/fp16/ --show
"""

from pathlib import Path
import argparse
import time
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

# Luna components
from bakery.adapters.openvino.model_repository import ModelRepository
from bakery.adapters.openvino.inference_engine import InferenceEngine
from bakery.adapters.openvino.preprocessing import preprocess_with_metadata
from bakery.adapters.openvino.postprocessing import postprocess_segmentation, postprocess_pose
from bakery.annotators.disney_annotator import DisneyAnnotator, RenderConfig
from bakery.core.entities.model_config import ModelType


def convert_to_supervision_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    masks: np.ndarray,
    ratio: float,
    pad_w: float,
    pad_h: float,
    orig_width: int,
    orig_height: int
) -> sv.Detections:
    """
    Convert outputs to supervision.Detections format.

    Copied from run_lens.origin.py - this is the working version.
    """
    if len(boxes) == 0:
        return sv.Detections.empty()

    # 1. Undo letterbox from boxes
    boxes_orig = boxes.copy()
    boxes_orig[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio  # x1, x2
    boxes_orig[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio  # y1, y2

    # Clamp to original dimensions
    boxes_orig[:, [0, 2]] = np.clip(boxes_orig[:, [0, 2]], 0, orig_width)
    boxes_orig[:, [1, 3]] = np.clip(boxes_orig[:, [1, 3]], 0, orig_height)

    # 2. Undo letterbox from masks
    masks_orig = []
    for mask in masks:
        # Remove padding from mask (mask is in input_shape coords)
        mask_h, mask_w = mask.shape
        pad_top = int(pad_h)
        pad_bottom = int(pad_h)
        pad_left = int(pad_w)
        pad_right = int(pad_w)

        # Crop mask to remove padding
        mask_unpadded = mask[pad_top:mask_h-pad_bottom, pad_left:mask_w-pad_right]

        # Resize to original frame dimensions
        if mask_unpadded.shape[0] > 0 and mask_unpadded.shape[1] > 0:
            mask_resized = cv2.resize(
                mask_unpadded,
                (orig_width, orig_height),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            # Fallback: resize directly
            mask_resized = cv2.resize(
                mask,
                (orig_width, orig_height),
                interpolation=cv2.INTER_LINEAR
            )

        # Threshold for binary mask
        mask_bin = (mask_resized > 0.5).astype(bool)
        masks_orig.append(mask_bin)

    masks_orig = np.array(masks_orig)

    # 3. Create sv.Detections
    detections = sv.Detections(
        xyxy=boxes_orig,
        confidence=scores,
        class_id=class_ids.astype(int),
        mask=masks_orig
    )

    return detections


def convert_to_supervision_keypoints(
    keypoints: np.ndarray,
    ratio: float,
    pad_w: float,
    pad_h: float,
    orig_width: int,
    orig_height: int,
    keypoint_conf_threshold: float = 0.5
) -> sv.KeyPoints:
    """
    Convert keypoints to supervision.KeyPoints format.

    Copied from run_lens.origin.py - this is the working version.
    """
    if len(keypoints) == 0:
        return sv.KeyPoints.empty()

    # Extract xy and confidence
    xy = keypoints[:, :, :2].copy()  # [N, 17, 2]
    confidence = keypoints[:, :, 2].copy()   # [N, 17]

    # Create mask for low confidence keypoints
    low_conf_mask = confidence < keypoint_conf_threshold

    # Undo letterbox transformation
    xy[:, :, 0] = (xy[:, :, 0] - pad_w) / ratio  # x
    xy[:, :, 1] = (xy[:, :, 1] - pad_h) / ratio  # y

    # Clamp to original dimensions
    xy[:, :, 0] = np.clip(xy[:, :, 0], 0, orig_width)
    xy[:, :, 1] = np.clip(xy[:, :, 1], 0, orig_height)

    # Mark low confidence keypoints as [0, 0]
    xy[low_conf_mask] = 0
    confidence[low_conf_mask] = 0

    # Create sv.KeyPoints
    return sv.KeyPoints(
        xy=xy.astype(np.float32),
        confidence=confidence.astype(np.float32),
        class_id=np.zeros(len(keypoints), dtype=int)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Luna components with original flow (validation script)"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing OpenVINO models"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/output_origin.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Filter class IDs (e.g., 0 for person)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("       Bakery - Luna Origin (Validation Script)")
    print("=" * 60)

    # Validate paths
    video_path = Path(args.video)
    models_dir = Path(args.models_dir)
    output_path = Path(args.output)

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return

    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover models
    print(f"\nDiscovering models in: {models_dir}")
    repo = ModelRepository(models_dir)
    models = repo.discover_models()

    if not models:
        print("Error: No models found")
        return

    # Find segmentation and pose models
    seg_models = [m for m in models if m.model_type == ModelType.SEGMENTATION]
    pose_models = [m for m in models if m.model_type == ModelType.POSE]

    if not seg_models:
        print("Error: No segmentation models found")
        return

    # Use first available models
    seg_model = seg_models[0]
    pose_model = pose_models[0] if pose_models else None

    print(f"Segmentation: {seg_model.model_path.stem} ({seg_model.resolution}px)")
    if pose_model:
        print(f"Pose: {pose_model.model_path.stem} ({pose_model.resolution}px)")

    # Create inference engines (InferenceEngine compiles automatically in __init__)
    print("\nCompiling models...")
    seg_engine = InferenceEngine(seg_model)
    print(f"  Segmentation: {seg_engine.device}")

    pose_engine = None
    if pose_model:
        pose_engine = InferenceEngine(pose_model)
        print(f"  Pose: {pose_engine.device}")

    # Get input shapes (get_input_shape returns (h, w) tuple)
    seg_h, seg_w = seg_engine.get_input_shape()

    pose_h, pose_w = None, None
    if pose_engine:
        pose_h, pose_w = pose_engine.get_input_shape()

    # Create annotator
    render_config = RenderConfig(
        bw_darkness=0.6,
        lens_brightness=1.1,
        spotlight_brightness=1.2,
    )
    annotator = DisneyAnnotator(render_config, enable_pose=(pose_engine is not None))

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {frame_width}x{frame_height} @ {fps:.1f} FPS, {total_frames} frames")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    print(f"\nProcessing video...")
    if args.show:
        print("Live preview enabled (press 'q' to quit)")

    start_time = time.time()
    frame_count = 0

    pbar = tqdm(total=total_frames, desc="Frames")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess for segmentation
            seg_tensor, seg_ratio, (seg_pad_w, seg_pad_h) = preprocess_with_metadata(
                frame, (seg_h, seg_w)
            )

            # Run segmentation inference
            seg_outputs = seg_engine.infer(seg_tensor)
            output_keys = list(seg_outputs.keys())

            # YOLO seg has 2 outputs: boxes and masks
            output_boxes = seg_outputs[output_keys[0]]
            output_masks = seg_outputs[output_keys[1]] if len(output_keys) > 1 else None

            # Postprocess segmentation
            if output_masks is not None:
                boxes, scores, class_ids, masks = postprocess_segmentation(
                    output_boxes, output_masks, (seg_h, seg_w),
                    conf_threshold=args.confidence
                )
            else:
                boxes, scores, class_ids, masks = np.array([]), np.array([]), np.array([]), np.array([])

            # Convert to supervision detections (using original function)
            detections = convert_to_supervision_detections(
                boxes, scores, class_ids, masks,
                seg_ratio, seg_pad_w, seg_pad_h,
                frame_width, frame_height
            )

            # Filter by classes if specified
            if args.classes is not None and len(detections) > 0:
                mask = np.isin(detections.class_id, args.classes)
                detections = detections[mask]

            # Run pose inference if available
            keypoints = None
            if pose_engine:
                # Preprocess for pose (may be different resolution)
                if (pose_h, pose_w) != (seg_h, seg_w):
                    pose_tensor, pose_ratio, (pose_pad_w, pose_pad_h) = preprocess_with_metadata(
                        frame, (pose_h, pose_w)
                    )
                else:
                    pose_tensor = seg_tensor
                    pose_ratio = seg_ratio
                    pose_pad_w, pose_pad_h = seg_pad_w, seg_pad_h

                # Run pose inference
                pose_outputs = pose_engine.infer(pose_tensor)
                pose_output_keys = list(pose_outputs.keys())
                pose_output = pose_outputs[pose_output_keys[0]]

                # Postprocess pose
                pose_boxes, pose_scores, pose_class_ids, pose_keypoints = postprocess_pose(
                    pose_output, (pose_h, pose_w),
                    conf_threshold=args.confidence
                )

                # Convert to supervision keypoints
                if len(pose_keypoints) > 0:
                    keypoints = convert_to_supervision_keypoints(
                        pose_keypoints, pose_ratio, pose_pad_w, pose_pad_h,
                        frame_width, frame_height,
                        keypoint_conf_threshold=args.confidence
                    )

            # Annotate frame using Luna's DisneyAnnotator
            annotated = annotator.annotate(
                frame=frame,
                detections=detections,
                keypoints=keypoints
            )

            # Write frame
            writer.write(annotated)

            # Show preview if enabled
            if args.show:
                cv2.imshow("Luna Origin", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break

            frame_count += 1
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    fps_actual = frame_count / elapsed if elapsed > 0 else 0

    print(f"\n" + "=" * 60)
    print(f"Completed!")
    print(f"  Frames: {frame_count}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  FPS: {fps_actual:.2f}")
    print(f"  Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
