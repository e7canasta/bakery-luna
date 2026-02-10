# C4 Model View: Origin Architecture - run_lens.origin.py

## Container Diagram - Origin

```mermaid
C4Container
    title Origin Architecture - Single Stream run_lens.origin.py (~1,600 LOC)

    Person(user, "User")

    Container_Boundary(current, "Origin System") {
        Container(run_lens, "run_lens.origin.py", "Python Script", "Sequential inference pipeline (Seg + Pose)")
    }

    System_Ext(video, "Video Files")
    System_Ext(openvino, "OpenVINO Runtime")
    System_Ext(supervision, "Supervision Library")
    System_Ext(models, "Model Repository")

    Rel(user, run_lens, "Ejecuta", "CLI: uv run run_lens.origin.py")
    Rel(run_lens, video, "Lee frames", "cv2.VideoCapture")
    Rel(run_lens, models, "Descubre modelos", "Explicit paths")
    Rel(run_lens, openvino, "Inferencia", "Sequential (Seg -> Pose)")
    Rel(run_lens, supervision, "Anota", "Disney Style")
```

## Component Diagram - Internal Structure of run_lens.origin.py

```mermaid
C4Component
    title run_lens.origin.py - Component Breakdown

    Container_Boundary(run_lens, "run_lens.origin.py") {
        Component(cli, "CLI Parser", "argparse", "Basic arguments (model size, res, focus)")
        Component(discovery, "Model Discovery", "Python", "Simple path matching (no fallback)")
        Component(preprocess, "Preprocessing", "Python/OpenCV", "Letterbox, Focus Lens crop")
        Component(inference, "Inference Loop", "OpenVINO", "Sequential: Frame -> Seg -> Pose")
        Component(post_seg, "Seg Postproc", "Python", "NMS, Mask processing")
        Component(post_pose, "Pose Postproc", "Python", "Keypoint extraction")
        Component(mapping, "Coord Mapping", "Python", "Map crop coords to full frame")
        Component(vis, "Visualization", "Supervision", "Halo, Glow, Translucent overlays")
        Component(io, "Video I/O", "Supervision", "VideoSink")
    }

    Rel(cli, discovery, "Configures")
    Rel(discovery, inference, "Loads models")
    Rel(inference, preprocess, "Prepares input")
    Rel(preprocess, inference, "Input tensor")
    Rel(inference, post_seg, "Raw Seg Output")
    Rel(inference, post_pose, "Raw Pose Output")
    Rel(post_seg, mapping, "Local Detections")
    Rel(post_pose, mapping, "Local Keypoints")
    Rel(mapping, vis, "Global Detections/Keypoints")
    Rel(vis, io, "Annotated Frame")
```

## Functionality Mapping: Origin vs. Extended

This table maps the functionality from the lighter `run_lens.origin.py` to the more complex `run_lens.py.sample` (Extended/Hybrid).

| Feature | Origin (`run_lens.origin.py`) | Extended (`run_lens.py.sample`) | Change Type |
| :--- | :--- | :--- | :--- |
| **Model Discovery** | `discover_segmentation_models`<br>`discover_pose_models`<br>Strict matching. Fails if exact model not found. | `discover_models_dual`<br>Includes **Fallbacks** (INT8 -> FP16, Resolution -> Lower Res). | ✨ Enhancement |
| **Scheduling** | **Sequential / Every Frame**<br>Runs Segmentation AND Pose for every single frame. | **Smart Scheduling**<br>`seg_interval`: Seg every N frames, Pose every frame.<br>Reuses cached seg masks. | 🚀 Optimization |
| **Preprocessing** | Computed every time in loop.<br>`preprocess_with_metadata` | **`PreprocessCache` Class**<br>Caches tensors. If `seg_res == pose_res`, computes once and reuses tensor. | 🚀 Optimization |
| **Keypoint Filtering** | **Basic**<br>Confidence thresholding only. | **Dynamic Strategy** (`filter_keypoints_by_masks_dynamic`)<br>Checks bbox overlap. Switches between "nearest centroid" and "bbox IoU". | 🧠 Logic |
| **Device Support** | Single `--device` (default GPU).<br>Both models run on same device. | **Heterogeneous Compute**<br>`--seg-device` and `--pose-device`.<br>Can offload Pose to CPU while Seg runs on GPU. | ⚙️ Config |
| **Inference Function** | `run_inference_fp16_segmentation_sv_lens` | `run_inference_fp16_segmentation_sv_lens` (Overloaded)<br>Added args: `seg_interval`, `pose_device`, `keypoint_match_strategy`. | 🔄 Refactor |
| **Visualization** | Disney/Roger Rabbit style.<br>Halo, Glow, Translucent UI. | Same core style, but integrated with scheduler (shows cached segmentation visually same as real). | 🎨 UI |
| **Code Structure** | Functional script.<br>`main()` calls `run_inference...` | Functional + OOP.<br>Introduces `PreprocessCache` class to manage state. | 🏗️ Architecture |

## Reverse Engineering Summary

1.  **Core Logic Preserved**: The fundamental "Focus Lens" strategy (crop -> inference -> map back) and the "Disney" visualization style are identical in both versions.
2.  **Performance Layer Added**: The Extended version wraps the Core Logic with performance optimizations:
    -   **Caching**: To avoid resizing/normalizing the same image twice.
    -   **Scheduling**: To skip heavy segmentation models on intermediate frames.
    -   **Resilience**: To find "good enough" models if the exact requested one is missing.
3.  **Logic Sophistication**: The Extended version solves the "Multiple Person" problem in pose estimation by adding the Dynamic Keypoint Filtering, which links keypoints to specific segmentation masks more accurately than simple distance.
