"""
Bakery CLI - Validate Command (Single Model Validation)
"""
import typer
from pathlib import Path
from typing import Optional, List, Tuple
import time
import cv2
import numpy as np
from tqdm import tqdm
from rich.console import Console

# Bakery imports
from bakery.annotators.disney_annotator import DisneyAnnotator, RenderConfig
from bakery.core.entities import Frame

# New Architecture Imports
from bakery_catalog import ModelRepository
from bakery_runtime import ModelInstance, Device as RuntimeDevice

from bakery_cli.shared.video import open_video_source, create_video_writer
from bakery_cli.shared.supervision_utils import convert_to_supervision_detections, convert_to_supervision_keypoints

console = Console()

def run(
    video: str = typer.Option(..., help="Video file or camera index"),
    models_dir: Path = typer.Option(..., help="Directory containing OpenVINO models"),
    output: Optional[Path] = typer.Option(None, help="Output video path (default: results/output_validate.mp4)"),
    confidence: float = typer.Option(0.25, help="Confidence threshold"),
    show: bool = typer.Option(False, help="Show live preview"),
    seg_model: Optional[Path] = typer.Option(None, help="Specific seg model path"),
    pose_model: Optional[Path] = typer.Option(None, help="Specific pose model path"),
):
    """
    Validate models directly (bypassing Luna complex pipeline).
    """
    if output is None:
        output = Path("results") / "output_validate.mp4"

    console.rule("[bold magenta]🔍 Bakery - Model Validation[/bold magenta]")

    # 1. Discover Models (Simple Logic)
    repo = ModelRepository(models_dir)
    
    seg_instance = None
    pose_instance = None

    # Load Seg
    if seg_model:
        # TODO: Implement specific path loading properly in runtime factory if needed
        pass 
    else:
        # Just pick first available segmentation model
        # Using the same naive approach as run.py for now
        # Ideally: repo.get_first(type="segmentation")
        # Creating a helper here
        defaults = ["yolo11n-seg", "yolo26n-seg", "yolo8n-seg"]
        for name in defaults:
            info = repo.get(name, 640, "fp16") # Try common
            if info:
                seg_instance = ModelInstance.from_info(info, device=RuntimeDevice.GPU, confidence=confidence)
                console.print(f"   ✅ Segmentation: {name}")
                break
    
    # Load Pose
    if pose_model:
        pass
    else:
        defaults = ["yolo11n-pose", "yolo26n-pose"]
        for name in defaults:
            info = repo.get(name, 640, "fp16")
            if info:
                pose_instance = ModelInstance.from_info(info, device=RuntimeDevice.GPU, confidence=confidence)
                console.print(f"   ✅ Pose: {name}")
                break

    if not seg_instance:
        console.print("[red]❌ No segmentation model found (checked common names)[/red]")
        raise typer.Exit(1)

    # 2. Open Video
    cap, width, height, fps, total_frames = open_video_source(video)
    writer = create_video_writer(output, width, height, fps)

    # 3. Annotator
    annotator = DisneyAnnotator(
        RenderConfig(bw_darkness=0.6, lens_brightness=1.1, spotlight_brightness=1.2), 
        enable_pose=(pose_instance is not None)
    )

    # 4. Loop
    console.print("\n🎬 Validating models...")
    
    pbar = tqdm(total=total_frames if total_frames > 0 else None, unit="frames")
    
    if show:
        cv2.namedWindow("Bakery Validate", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, image = cap.read()
            if not ret:
                break
            
            # Direct inference
            # 1. Seg
            seg_tensor, seg_meta = seg_instance.preprocess(image)
            seg_out = seg_instance.infer_tensor(seg_tensor)
            boxes, scores, class_ids, masks = seg_instance.postprocess(seg_out, seg_meta)
            
            detections = convert_to_supervision_detections(
                boxes, scores, class_ids, masks, seg_meta
            )

            # 2. Pose (if available)
            keypoints = None
            if pose_instance:
                pose_tensor, pose_meta = pose_instance.preprocess(image)
                pose_out = pose_instance.infer_tensor(pose_tensor)
                p_boxes, p_scores, p_ids, p_kpts = pose_instance.postprocess(pose_out, pose_meta)
                
                keypoints = convert_to_supervision_keypoints(p_kpts, pose_meta, confidence)

            # Annotate
            annotated = annotator.annotate(image, detections, keypoints)
            
            writer.write(annotated)
            if show:
                cv2.imshow("Bakery Validate", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()
    
    console.print(f"\n✨ Validation complete. Output: {output}") 
