"""
Bakery CLI - Run Command
"""
import typer
from pathlib import Path
from typing import Optional, List, Tuple
import datetime
import time
import cv2
from tqdm import tqdm
from rich.console import Console

# Bakery imports
from bakery.core.entities import Frame, PipelineConfig, FocusLensConfig, ModelConfig, ModelType, Device, Precision
from bakery.pipeline.dual_model_pipeline import DualModelPipeline
from bakery.annotators.disney_annotator import DisneyAnnotator, RenderConfig
from bakery.utils.focus_lens import crop_info_to_tuple

# New Architecture Imports
from bakery_catalog import ModelRepository
from bakery_runtime import ModelInstance, Device as RuntimeDevice
from bakery_runtime.filtering import FilterPolicy

from bakery_cli.shared.video import open_video_source, create_video_writer

console = Console()

def discover_models(models_dir: Path, seg_model_path: Optional[Path], pose_model_path: Optional[Path], 
                   yolo_version: Optional[str], seg_device: Optional[str], pose_device: Optional[str], confidence: float,
                   seg_size: Optional[str], pose_size: Optional[str], resolution: Optional[int]) -> Tuple[ModelInstance, ModelInstance]:
    """
    Discover segmentation and pose models in directory using bakery-catalog.
    """
    console.print(f"\n🔍 Discovering models in: {models_dir}")
    repo = ModelRepository(models_dir)

    # Common configs
    common_resolutions = [640, 320, 512, 480, 256]
    common_versions = ["11", "26", "8"]

    def find_and_build(m_type, device_arg, size_arg, specific_path=None):
        # 1. Specific path (Override)
        if specific_path:
             # Basic fallback or specific path logic
             pass
        
        # 2. Discovery
        target_sizes = [size_arg] if size_arg else ["n", "s", "m", "l", "x"]
        target_resolutions = [resolution] if resolution else common_resolutions

        for ver in common_versions:
            if yolo_version and yolo_version != ver:
                continue
            
            for size in target_sizes:
                name = f"yolo{ver}{size}-{m_type}"
                for res in target_resolutions:
                    # Try FP16
                    info = repo.get(name, res, "fp16")
                    if info:
                        device = RuntimeDevice(device_arg) if device_arg else RuntimeDevice.GPU
                        instance = ModelInstance.from_info(info, device=device, confidence=confidence)
                        console.print(f"   ✅ Found {m_type}: {name} ({res}px) | Device: {instance.active_device}")
                        return instance
        return None

    console.print("   Searching for compatible models...")
    seg_instance = find_and_build("seg", seg_device, seg_size, seg_model_path)
    pose_instance = find_and_build("pose", pose_device, pose_size, pose_model_path)

    if not seg_instance or not pose_instance:
         console.print("[red]❌ Could not find standard named models (e.g., yolo11n-seg_640_fp16).[/red]")
         console.print("   Ensure your models directory follows the convention: {name}_{res}_{format}.xml")
         raise typer.Exit(code=1)

    return seg_instance, pose_instance


def create_pipeline_config(seg_instance: ModelInstance, pose_instance: ModelInstance, 
                          seg_interval: int, confidence: float, classes: Optional[List[int]]) -> PipelineConfig:
    """Create PipelineConfig from instances (Shim for legacy config)."""
    
    seg_config = ModelConfig(
        model_path=seg_instance.info.model_path,
        model_type=ModelType.SEGMENTATION,
        resolution=seg_instance.resolution,
        device=Device(seg_instance.device.value),
        precision=Precision(seg_instance.info.precision.value),
        confidence=confidence
    )

    pose_config = ModelConfig(
        model_path=pose_instance.info.model_path,
        model_type=ModelType.POSE,
        resolution=pose_instance.resolution,
        device=Device(pose_instance.device.value),
        precision=Precision(pose_instance.info.precision.value),
        confidence=confidence
    )

    return PipelineConfig(
        segmentation=seg_config,
        pose=pose_config,
        seg_interval=seg_interval,
        confidence_threshold=confidence,
        class_filter=classes,
    )


def run(
    video: str = typer.Option(..., help="Video file or camera index"),
    models_dir: Path = typer.Option(..., help="Directory containing OpenVINO models"),
    output: Optional[Path] = typer.Option(None, help="Output video path (default: results/output_luna.mp4)"),
    seg_interval: int = typer.Option(5, help="Run segmentation every N frames"),
    confidence: float = typer.Option(0.25, help="Global confidence threshold"),
    seg_confidence: Optional[float] = typer.Option(None, help="Confidence override for segmentation model"),
    pose_confidence: Optional[float] = typer.Option(None, help="Confidence override for pose model"),
    classes: Optional[List[int]] = typer.Option(None, help="Filter specific class IDs"),
    class_confidence: Optional[List[float]] = typer.Option(None, help="Per-class confidence (paired with --classes)"),
    keypoints: Optional[List[int]] = typer.Option(None, help="Keypoint indices to threshold"),
    keypoint_confidence: Optional[List[float]] = typer.Option(None, help="Per-keypoint confidence (paired with --keypoints)"),
    keypoint_confidence_all: Optional[float] = typer.Option(None, help="Blanket keypoint confidence threshold"),
    keypoint_min_visible: Optional[int] = typer.Option(None, help="Min visible keypoints to keep a skeleton"),
    show: bool = typer.Option(False, help="Show live preview"),
    focus_size: Optional[int] = typer.Option(None, help="Focus lens size"),
    focus_x: Optional[int] = typer.Option(None, help="Focus X position"),
    focus_y: Optional[int] = typer.Option(None, help="Focus Y position"),
    focus_strategy: str = typer.Option("zoom", help="Focus strategy (zoom/pad)"),
    adaptive: bool = typer.Option(False, help="Enable adaptive focus lens (shift strategy)"),
    allow_expand: bool = typer.Option(False, help="Allow adaptive lens to expand size"),
    edge_threshold: int = typer.Option(40, help="Pixels from edge to trigger shift"),
    shift_step: int = typer.Option(80, help="Max pixels to shift per update"),
    seg_device: str = typer.Option(None, help="Device for segmentation"),
    pose_device: str = typer.Option(None, help="Device for pose"),
    seg_model: Optional[Path] = typer.Option(None, help="Specific seg model path"),
    pose_model: Optional[Path] = typer.Option(None, help="Specific pose model path"),
    yolo_version: str = typer.Option(None, help="Filter by YOLO version"),
    seg_size: Optional[str] = typer.Option(None, help="Model size for segmentation (n, s, m, l, x)"),
    pose_size: Optional[str] = typer.Option(None, help="Model size for pose (n, s, m, l, x)"),
    resolution: Optional[int] = typer.Option(None, help="Force input resolution (e.g. 640)"),
):
    """
    Run the Bakery Vision Pipeline (Luna).
    """
    console.rule("[bold magenta]🌙 Bakery Vision Pipeline - Luna[/bold magenta]")
    
    # 1. Discover Models
    seg_instance, pose_instance = discover_models(
        models_dir, seg_model, pose_model, yolo_version, seg_device, pose_device, confidence,
        seg_size, pose_size, resolution
    )

    if output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        seg_name = seg_instance.info.model_name
        seg_res = seg_instance.resolution
        pose_name = pose_instance.info.model_name
        pose_res = pose_instance.resolution
        
        output = Path("results") / f"luna_{timestamp}_{seg_name}_{seg_res}_{pose_name}_{pose_res}.mp4"
    
    # Ensure parent dir exists
    if not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)

    # 2. Config & Pipeline
    console.print("\n🔧 Creating inference pipeline...")
    pipeline_config = create_pipeline_config(seg_instance, pose_instance, seg_interval, confidence, classes)

    # Build FilterPolicy from CLI args
    filter_policy = FilterPolicy.from_cli(
        confidence=confidence,
        seg_confidence=seg_confidence,
        pose_confidence=pose_confidence,
        classes=classes,
        class_confidence=class_confidence,
        keypoints=keypoints,
        keypoint_confidence=keypoint_confidence,
        keypoint_confidence_all=keypoint_confidence_all,
        keypoint_min_visible=keypoint_min_visible,
    )
    
    # Print filter summary
    if filter_policy.has_class_filter:
        console.print(f"   🎯 Class filter: {filter_policy.classes}")
    if filter_policy.has_per_class_confidence:
        console.print(f"   🎯 Per-class confidence: {filter_policy.per_class_confidence}")
    if filter_policy.has_keypoint_filter:
        console.print(f"   🦴 Keypoint filter active")
        if filter_policy.per_keypoint_confidence:
            console.print(f"      Per-keypoint: {filter_policy.per_keypoint_confidence}")
        if filter_policy.keypoint_min_visible:
            console.print(f"      Min visible: {filter_policy.keypoint_min_visible}")
    
    focus_lens_config = None
    if focus_size:
        focus_lens_config = FocusLensConfig(
            focus_size=focus_size, 
            focus_x=focus_x, 
            focus_y=focus_y, 
            strategy=focus_strategy,
            adaptive=adaptive,
            edge_threshold=edge_threshold,
            shift_step=shift_step,
            allow_expand=allow_expand
        )
        mode_str = "Adaptive Shift" if adaptive else "Static"
        if adaptive and allow_expand:
            mode_str += " + Expand"
        console.print(f"   🔍 Focus Lens: {focus_size}px ({focus_strategy}) | Mode: {mode_str}")

    pipeline = DualModelPipeline(seg_instance, pose_instance, pipeline_config, focus_lens_config, filter_policy)
    console.print(f"   ✅ Pipeline ready (seg_interval={seg_interval})")

    # 3. Annotator
    annotator = DisneyAnnotator(
        RenderConfig(bw_darkness=0.6, lens_brightness=1.2, spotlight_brightness=1.2), 
        enable_pose=True
    )

    # 4. Open Video
    cap, width, height, fps, total_frames = open_video_source(video)
    writer = create_video_writer(output, width, height, fps)

    # 5. Process Loop
    console.print("\n🎬 Processing video...")
    
    frame_id = 0
    pbar = tqdm(total=total_frames if total_frames > 0 else None, unit="frames", desc="Processing")
    
    if show:
        cv2.namedWindow("Luna Pipeline", cv2.WINDOW_NORMAL)

    start_time = time.time()
    try:
        while True:
            ret, image = cap.read()
            if not ret:
                break

            frame = Frame.from_array(image, frame_id=frame_id)
            
            # Helper to allow interrupting safely
            if frame_id % 10 == 0 and not pbar.disable: 
                 pass # check something?

            segmentation, pose_estimation = pipeline.process_frame(frame)

            detections = segmentation.to_supervision()
            keypoints = pose_estimation.to_supervision()
            
            crop_info = pipeline.get_crop_info()
            focus_region = crop_info_to_tuple(crop_info)

            annotated = annotator.annotate(image, detections, keypoints, focus_region)

            writer.write(annotated)
            if show:
                cv2.imshow("Luna Pipeline", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    console.print("\n⏸️  Stopped by user")
                    break

            frame_id += 1
            pbar.update(1)

    except KeyboardInterrupt:
        console.print("\n⏸️  Interrupted by user")
    finally:
        pbar.close()
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    duration = time.time() - start_time
    
    # Metrics
    metrics = pipeline.get_metrics()
    console.print("\n📊 Performance Metrics")
    console.print(f"Total frames: {metrics.total_frames}")
    console.print(f"FPS: {metrics.total_frames / duration:.2f}")
    console.print(f"Output saved to: {output}")

