#!/usr/bin/env python3
"""
Bakery Vision Pipeline - Luna 🌙
Demo script showcasing the complete Luna pipeline.

This script demonstrates:
- Model discovery with ModelRepository
- Dual-model inference with DualModelPipeline
- Disney/Roger Rabbit aesthetic rendering
- Performance metrics tracking

Usage:
    uv run run_luna.py --video videos/sample.mp4 --models-dir exports/fp16/
    uv run run_luna.py --video 0 --models-dir exports/fp16/  # Webcam

Requirements:
    - OpenVINO models in exports/fp16/ directory
    - Video file or webcam access
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

# Bakery imports
from bakery.core.entities import Frame, PipelineConfig, FocusLensConfig
from bakery.adapters.openvino.model_repository import ModelRepository
from bakery.adapters.openvino.inference_engine import InferenceEngine
from bakery.pipeline.dual_model_pipeline import DualModelPipeline
from bakery.annotators.disney_annotator import DisneyAnnotator, RenderConfig
from bakery.core.entities.model_config import ModelType
from bakery.utils.focus_lens import crop_info_to_tuple


def discover_models(models_dir: Path, args) -> tuple:
    """
    Discover segmentation and pose models in directory, or use specific model paths.

    Args:
        models_dir: Directory containing OpenVINO models
        args: Command-line arguments (may contain --seg-model and --pose-model)

    Returns:
        Tuple of (segmentation_model, pose_model) configs
    """
    from bakery.core.entities.model_config import ModelConfig, Device, Precision

    seg_model = None
    pose_model = None

    # Check for specific model paths first
    if args.seg_model:
        print(f"\n📦 Using specific segmentation model: {args.seg_model}")
        if not args.seg_model.exists():
            print(f"❌ Segmentation model not found: {args.seg_model}")
            sys.exit(1)
        # Create ModelConfig from specific path
        seg_model = _create_model_config_from_path(args.seg_model, ModelType.SEGMENTATION)

    if args.pose_model:
        print(f"🦴 Using specific pose model: {args.pose_model}")
        if not args.pose_model.exists():
            print(f"❌ Pose model not found: {args.pose_model}")
            sys.exit(1)
        # Create ModelConfig from specific path
        pose_model = _create_model_config_from_path(args.pose_model, ModelType.POSE)

    # Discover remaining models from directory if needed
    if not seg_model or not pose_model:
        print(f"\n🔍 Discovering models in: {models_dir}")

        repository = ModelRepository(models_dir)
        # Filter by YOLO version if specified
        yolo_version = getattr(args, 'yolo_version', None)
        if yolo_version:
            print(f"   🎯 Filtering by YOLO version: {yolo_version}")
        models = repository.discover_models(yolo_version=yolo_version)

        if not models and not (seg_model and pose_model):
            print(f"❌ No models found in {models_dir}")
            if yolo_version:
                print(f"   (with YOLO version filter: {yolo_version})")
            sys.exit(1)

        print(f"✅ Found {len(models)} model(s)")

        # Get models by type if not already specified
        if not seg_model:
            seg_model = repository.get_model_by_type(ModelType.SEGMENTATION, yolo_version=yolo_version)
        if not pose_model:
            pose_model = repository.get_model_by_type(ModelType.POSE, yolo_version=yolo_version)

    # Print model info
    if seg_model:
        print(f"   📦 Segmentation: {seg_model.model_path.name} ({seg_model.resolution}px, {seg_model.precision.value})")
    else:
        print("   ⚠️  No segmentation model found")

    if pose_model:
        print(f"   🦴 Pose: {pose_model.model_path.name} ({pose_model.resolution}px, {pose_model.precision.value})")
    else:
        print("   ⚠️  No pose model found")

    if not seg_model or not pose_model:
        print("\n❌ Both segmentation and pose models are required")
        sys.exit(1)

    # Override devices if specified
    if args.seg_device:
        seg_model.device = Device(args.seg_device)
        print(f"   ⚙️  Segmentation device override: {args.seg_device}")

    if args.pose_device:
        pose_model.device = Device(args.pose_device)
        print(f"   ⚙️  Pose device override: {args.pose_device}")

    return seg_model, pose_model


def _create_model_config_from_path(model_path: Path, model_type: ModelType):
    """
    Create ModelConfig from a specific model path.

    Args:
        model_path: Path to .xml model file
        model_type: Type of model (SEGMENTATION or POSE)

    Returns:
        ModelConfig instance
    """
    import openvino as ov
    from bakery.core.entities.model_config import ModelConfig, Device, Precision

    core = ov.Core()
    model = core.read_model(str(model_path))

    # Extract resolution from input shape
    input_shape = tuple(model.input(0).shape)
    resolution = input_shape[2]  # Assuming [N, C, H, W]

    # Detect precision from path
    path_str = str(model_path).lower()
    if "int8" in path_str:
        precision = Precision.INT8
    elif "fp16" in path_str:
        precision = Precision.FP16
    else:
        precision = Precision.FP32

    # Default device based on precision
    # INT8 -> CPU (for VNNI), FP16/FP32 -> GPU
    device = Device.CPU if precision == Precision.INT8 else Device.GPU

    return ModelConfig(
        model_path=model_path,
        model_type=model_type,
        resolution=resolution,
        device=device,
        precision=precision,
        confidence=0.25
    )


def create_pipeline(seg_model, pose_model, args) -> DualModelPipeline:
    """
    Create dual-model inference pipeline.

    Args:
        seg_model: Segmentation ModelConfig
        pose_model: Pose ModelConfig
        args: Command-line arguments

    Returns:
        Configured DualModelPipeline
    """
    print("\n🔧 Creating inference pipeline...")

    # Update model configs with user settings
    seg_model.confidence = args.confidence
    pose_model.confidence = args.confidence

    # Create inference engines
    print("   Compiling segmentation model...")
    seg_engine = InferenceEngine(seg_model)
    print(f"   ✅ Segmentation engine ready on {seg_engine.get_device()}")

    print("   Compiling pose model...")
    pose_engine = InferenceEngine(pose_model)
    print(f"   ✅ Pose engine ready on {pose_engine.get_device()}")

    # Create pipeline config
    pipeline_config = PipelineConfig(
        segmentation=seg_model,
        pose=pose_model,
        seg_interval=args.seg_interval,
        confidence_threshold=args.confidence,
        class_filter=args.classes,
    )

    # Create Focus Lens config if specified
    focus_lens_config = None
    if args.focus_size is not None:
        focus_lens_config = FocusLensConfig(
            focus_size=args.focus_size,
            focus_x=args.focus_x,
            focus_y=args.focus_y,
            strategy=args.focus_strategy
        )
        position = "centered" if focus_lens_config.is_centered else f"({args.focus_x}, {args.focus_y})"
        print(f"   🔍 Focus Lens: {args.focus_size}px {position} ({args.focus_strategy})")

    # Create pipeline
    pipeline = DualModelPipeline(seg_engine, pose_engine, pipeline_config, focus_lens_config)

    print(f"   ✅ Pipeline configured (seg_interval={args.seg_interval})")

    # Print optimization info
    if pipeline_config.is_same_resolution:
        print(f"   ⚡ Preprocessing cache optimization enabled (both models use {seg_model.resolution}px)")
    else:
        print(f"   ℹ️  Different resolutions: seg={seg_model.resolution}px, pose={pose_model.resolution}px")

    return pipeline


def create_annotator(args) -> DisneyAnnotator:
    """
    Create Disney-style annotator.

    Args:
        args: Command-line arguments

    Returns:
        Configured DisneyAnnotator
    """
    render_config = RenderConfig(
        bw_darkness=0.6,  # Darken B&W world to 60% of original
        lens_brightness=1.2,  # Brighten focus lens by 20%
        spotlight_brightness=1.2,  # Brighten detected objects by 20%
    )

    return DisneyAnnotator(render_config, enable_pose=True)


def open_video_source(video_path: str):
    """
    Open video file or camera.

    Args:
        video_path: Path to video file or camera index (0, 1, etc.)

    Returns:
        cv2.VideoCapture object
    """
    # Try to parse as camera index
    try:
        camera_idx = int(video_path)
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            print(f"\n📹 Opened camera {camera_idx}")
            return cap
    except ValueError:
        pass

    # Try as file path
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        sys.exit(1)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n📹 Video opened: {Path(video_path).name}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")

    return cap


def create_video_writer(output_path: Path, width: int, height: int, fps: int):
    """
    Create video writer for output.

    Args:
        output_path: Output video path
        width: Frame width
        height: Frame height
        fps: Frames per second

    Returns:
        cv2.VideoWriter object
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"❌ Failed to create video writer: {output_path}")
        sys.exit(1)

    print(f"\n💾 Output: {output_path}")

    return writer


def process_video(cap, writer, pipeline: DualModelPipeline, annotator: DisneyAnnotator, args):
    """
    Process video through pipeline.

    Args:
        cap: Video capture object
        writer: Video writer object
        pipeline: DualModelPipeline instance
        annotator: DisneyAnnotator instance
        args: Command-line arguments
    """
    print("\n🎬 Processing video...")
    print("=" * 60)

    frame_id = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Progress bar
    pbar = tqdm(total=total_frames if total_frames > 0 else None, unit="frames", desc="Processing")

    # Initialize display window if show is enabled
    if args.show:
        cv2.namedWindow("Luna Pipeline", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, image = cap.read()
            if not ret:
                break

            # Create Frame entity
            frame = Frame.from_array(image, frame_id=frame_id)

            # Run pipeline
            segmentation, pose_estimation = pipeline.process_frame(frame)

            # Convert entities to supervision format
            detections = segmentation.to_supervision()
            keypoints = pose_estimation.to_supervision()

            # Get crop info for focus lens visualization
            crop_info = pipeline.get_crop_info()
            focus_region = crop_info_to_tuple(crop_info)

            # Annotate frame
            annotated = annotator.annotate(
                frame=image,  # Use original full frame for annotation
                detections=detections,
                keypoints=keypoints,
                focus_region=focus_region
            )

            # Write frame
            writer.write(annotated)

            # Show preview (if enabled)
            if args.show:
                cv2.imshow("Luna Pipeline", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏸️  Stopped by user")
                    break

            frame_id += 1
            pbar.update(1)

    except KeyboardInterrupt:
        print("\n⏸️  Interrupted by user")

    finally:
        pbar.close()

    print(f"✅ Processed {frame_id} frames")


def print_metrics(pipeline: DualModelPipeline, frame_count: int, duration: float):
    """
    Print pipeline performance metrics.

    Args:
        pipeline: DualModelPipeline instance
        frame_count: Total frames processed
        duration: Total processing time in seconds
    """
    metrics = pipeline.get_metrics()

    print("\n📊 Performance Metrics")
    print("=" * 60)
    print(f"Total frames:         {metrics.total_frames}")
    print(f"Segmentation runs:    {metrics.seg_runs}")
    print(f"Pose runs:            {metrics.pose_runs}")
    print(f"Processing time:      {duration:.2f}s")
    print(f"Average FPS:          {frame_count / duration:.2f}")
    print()
    print(f"Seg efficiency:       {metrics.total_frames / max(metrics.seg_runs, 1):.1f}x (ran every {metrics.total_frames / max(metrics.seg_runs, 1):.1f} frames)")
    print(f"Pose efficiency:      {metrics.total_frames / max(metrics.pose_runs, 1):.1f}x (ran every frame)")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bakery Vision Pipeline - Luna 🌙",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with default settings
  uv run run_luna.py --video videos/sample.mp4 --models-dir exports/fp16/

  # Use webcam with custom settings
  uv run run_luna.py --video 0 --models-dir exports/fp16/ --seg-interval 3 --show

  # High confidence threshold, only detect persons
  uv run run_luna.py --video videos/sample.mp4 --models-dir exports/fp16/ --confidence 0.5 --classes 0

  # Output to custom directory
  uv run run_luna.py --video videos/sample.mp4 --models-dir exports/fp16/ --output results/my_output.mp4

  # With Focus Lens (crop-based inference) - centered 640x640
  uv run run_luna.py --video videos/sample.mp4 --models-dir exports/fp16/ --focus-size 640 --show

  # Focus Lens at specific position
  uv run run_luna.py --video videos/sample.mp4 --models-dir exports/fp16/ --focus-size 480 --focus-x 100 --focus-y 100

  # Focus Lens with pad strategy (for small frames)
  uv run run_luna.py --video videos/sample.mp4 --models-dir exports/fp16/ --focus-size 640 --focus-strategy pad

  # Hybrid CPU/GPU: Segmentation on GPU (FP16), Pose on CPU (INT8/VNNI)
  uv run run_luna.py --video videos/sample.mp4 --models-dir exports/fp16/ \\
      --seg-model exports/fp16/.../yolo11n-seg.xml --seg-device GPU \\
      --pose-model exports/int8/.../yolo11n-pose_int8.xml --pose-device CPU

  # Hybrid CPU/GPU: Segmentation on CPU (INT8/VNNI), Pose on GPU (FP16)
  uv run run_luna.py --video videos/sample.mp4 --models-dir exports/fp16/ \\
      --seg-model exports/int8/.../yolo11n-seg_int8.xml --seg-device CPU \\
      --pose-model exports/fp16/.../yolo11n-pose.xml --pose-device GPU

  # Filter by YOLO version (only use YOLO26 models)
  uv run run_luna.py --video videos/sample.mp4 --models-dir exports/fp16/ --yolo-version 26

  # Filter by YOLO version (only use YOLO11 models)
  uv run run_luna.py --video videos/sample.mp4 --models-dir exports/fp16/ --yolo-version 11
        """
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Video file path or camera index (0 for webcam)"
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Directory containing OpenVINO models (.xml files)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: results/output_luna.mp4)"
    )

    parser.add_argument(
        "--seg-interval",
        type=int,
        default=5,
        help="Run segmentation every N frames (default: 5)"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)"
    )

    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Filter specific class IDs (e.g., 0=person, 2=car)"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview during processing"
    )

    # Focus Lens arguments
    parser.add_argument(
        "--focus-size",
        type=int,
        default=None,
        help="Focus lens size in pixels (must be multiple of 80, e.g., 480, 640)"
    )

    parser.add_argument(
        "--focus-x",
        type=int,
        default=None,
        help="Focus X position (None = centered)"
    )

    parser.add_argument(
        "--focus-y",
        type=int,
        default=None,
        help="Focus Y position (None = centered)"
    )

    parser.add_argument(
        "--focus-strategy",
        type=str,
        choices=["zoom", "pad"],
        default="zoom",
        help="Strategy when frame < focus_size: 'zoom' (scale up) or 'pad' (add black borders)"
    )

    # Device and model override arguments for hybrid CPU/GPU inference
    parser.add_argument(
        "--seg-device",
        type=str,
        choices=["CPU", "GPU", "AUTO"],
        default=None,
        help="Device for segmentation model (default: from model config)"
    )

    parser.add_argument(
        "--pose-device",
        type=str,
        choices=["CPU", "GPU", "AUTO"],
        default=None,
        help="Device for pose model (default: from model config)"
    )

    parser.add_argument(
        "--seg-model",
        type=Path,
        default=None,
        help="Path to specific segmentation model .xml file (overrides --models-dir discovery)"
    )

    parser.add_argument(
        "--pose-model",
        type=Path,
        default=None,
        help="Path to specific pose model .xml file (overrides --models-dir discovery)"
    )

    parser.add_argument(
        "--yolo-version",
        type=str,
        default=None,
        help="Filter models by YOLO version (e.g., '26', '11', '8'). Filters by model name/path containing 'yolo{version}'"
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = Path("results") / "output_luna.mp4"

    # Print banner
    print("\n" + "=" * 60)
    print("🌙 Bakery Vision Pipeline - Luna".center(60))
    print("Dual-Model Inference with Disney Aesthetic".center(60))
    print("=" * 60)

    # Discover models (or use specific model paths if provided)
    seg_model, pose_model = discover_models(args.models_dir, args)

    # Create pipeline
    pipeline = create_pipeline(seg_model, pose_model, args)

    # Create annotator
    annotator = create_annotator(args)
    print("   ✅ DisneyAnnotator ready (8-layer rendering)")

    # Open video
    cap = open_video_source(args.video)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # Create video writer
    writer = create_video_writer(args.output, width, height, fps)

    # Process video
    import time
    start_time = time.time()

    process_video(cap, writer, pipeline, annotator, args)

    end_time = time.time()
    duration = end_time - start_time

    # Cleanup
    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # Print metrics
    metrics = pipeline.get_metrics()
    print_metrics(pipeline, metrics.total_frames, duration)

    print(f"\n✨ Done! Output saved to: {args.output}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
