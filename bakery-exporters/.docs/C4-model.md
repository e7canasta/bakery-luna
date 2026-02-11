# Bakery Exporters - C4 Model

## Context Diagram (Level 1)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Software System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐         ┌─────────────────────────┐  │
│  │  Data Scientist / ML │         │   ML Ops / Deployment   │  │
│  │     Engineer         │◄───────►│       Engineer          │  │
│  └──────────────────────┘         └─────────────────────────┘  │
│           ▲                              ▲                      │
│           │                              │                      │
│           │ Export commands              │ Export commands      │
│           │ (CLI/Python API)             │ (CLI/Python API)    │
│           │                              │                      │
│           ▼                              ▼                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │        Bakery Exporters Package                          │  │
│  │  (Model Export & Optimization)                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│           ▲                              │                      │
│           │                              ▼                      │
│           │                   ┌─────────────────────┐          │
│           └──────────────────►│  Bakery Catalog     │          │
│                               │  (Config, Paths)    │          │
│                               └─────────────────────┘          │
│                                       ▲                         │
│                                       │                         │
│                              Uses .env file                     │
│                                       │                         │
│                               ┌───────┴────────┐               │
│                               │                │                │
│                           ┌───────┐        ┌───────┐           │
│                           │ YOLO  │        │ONNX / │           │
│                           │Models │        │OpenVINO│          │
│                           │       │        │  IRs  │           │
│                           └───────┘        └───────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Container Diagram (Level 2)

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Bakery Exporters Package                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐  │
│  │   CLI Scripts    │  │  Python Package  │  │   Documentation    │  │
│  ├──────────────────┤  ├──────────────────┤  ├────────────────────┤  │
│  │ • export_*.py    │  │ bakery_exporters │  │ • Specification    │  │
│  │ • calibrate.py   │  │  • onnx.py       │  │ • ADRs             │  │
│  └──────────────────┘  │  • openvino.py   │  │ • USAGE.md         │  │
│         │              │  • calibration.py│  └────────────────────┘  │
│         │              │  • pipeline.py   │                           │
│         │              │  • cli.py        │                           │
│         │              └──────────────────┘                           │
│         │                      ▲ │                                    │
│         └──────────────────────┼─┘                                    │
│                                │                                      │
│                  Uses API & Utilities                                 │
│                                │                                      │
│         ┌──────────────────────┴───────────────────┐                 │
│         │                                          │                 │
│         ▼                                          ▼                 │
│  ┌────────────────────────────────────────┐  ┌──────────────────┐   │
│  │      Bakery Catalog (bakery/)          │  │  External Libs   │   │
│  ├────────────────────────────────────────┤  ├──────────────────┤   │
│  │  • config.py                           │  │ • ultralytics    │   │
│  │  • paths.py (ModelPath)                │  │ • openvino       │   │
│  │  • discovery.py                        │  │ • nncf           │   │
│  │  • constants.py                        │  │ • numpy          │   │
│  └────────────────────────────────────────┘  └──────────────────┘   │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

## Component Diagram (Level 3)

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Bakery Exporters Components                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ Export Pipeline (Orchestration)                                   │ │
│  ├───────────────────────────────────────────────────────────────────┤ │
│  │                                                                    │ │
│  │  ExportPipeline                                                   │ │
│  │  ├─ export(version, size, task, resolution, formats)             │ │
│  │  └─ export_batch(version, sizes, task, resolutions, formats)     │ │
│  │                                                                    │ │
│  └────────────────┬────────────────────────────┬─────────────────────┘ │
│                   │                            │                        │
│                   ▼                            ▼                        │
│  ┌──────────────────────────┐   ┌──────────────────────────────────┐  │
│  │ OnnxExporter             │   │ OpenVINOConverter                │  │
│  ├──────────────────────────┤   ├──────────────────────────────────┤  │
│  │                          │   │                                  │  │
│  │ • export()               │   │ • to_fp16()                      │  │
│  │   Exports YOLO models    │   │   (GPU inference)                │  │
│  │   to ONNX format         │   │                                  │  │
│  │                          │   │ • to_int8_synthetic()            │  │
│  └──────────┬───────────────┘   │   (CPU quick quantization)       │  │
│             │                   │                                  │  │
│             │ ONNX file         │ • to_int8_calibrated()           │  │
│             │                   │   (CPU production quantization)  │  │
│             ▼                   │                                  │  │
│  ┌──────────────────────────┐   └──────────────┬───────────────────┘  │
│  │ YOLO Model               │                  │                       │
│  │ (ultralytics)            │                  │ OpenVINO IR files     │
│  └──────────────────────────┘                  │                       │
│                                                ▼                       │
│                                   ┌──────────────────────────┐         │
│                                   │ Calibration Support      │         │
│                                   ├──────────────────────────┤         │
│                                   │                          │         │
│                                   │ CalibrationDataLoader    │         │
│                                   │ • __iter__()             │         │
│                                   │ • __len__()              │         │
│                                   │                          │         │
│                                   │ discover_onnx_models()   │         │
│                                   │                          │         │
│                                   └──────────────────────────┘         │
│                                            │                           │
│                                            ▼                           │
│                                   ┌──────────────────────────┐         │
│                                   │ NNCF Quantization        │         │
│                                   │ (nncf library)           │         │
│                                   └──────────────────────────┘         │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ CLI Support                                                       │ │
│  ├───────────────────────────────────────────────────────────────────┤ │
│  │                                                                    │ │
│  │  create_base_parser()      - Shared argument definitions          │ │
│  │  add_format_argument()     - Format selection                     │ │
│  │                                                                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
User Input (CLI or API)
│
├─ YOLO version, size, task
├─ Resolution
└─ Format(s) to export

           │
           ▼
    ┌─────────────────┐
    │ ExportPipeline  │
    │  .export()      │
    └────────┬────────┘
             │
             ├─ Validate inputs
             └─ Route to OnnxExporter
                        │
                        ▼
             ┌──────────────────────┐
             │  OnnxExporter        │
             │  .export()           │
             └────────┬─────────────┘
                      │
         ┌────────────┴─────────────┐
         │ Load YOLO model          │
         │ Export to ONNX           │
         │ Move to models_dir       │
         └────────┬─────────────────┘
                  │
         ┌────────▼─────────────────┐
         │ models/{name}/{res}/     │
         │ onnx/model.onnx          │
         └────────┬─────────────────┘
                  │
        ┌─────────┴──────────┬──────────────┐
        │                    │              │
        ▼ (if FP16)          ▼ (if INT8)    ▼ (if INT8_CAL)
    ┌───────────────┐    ┌──────────────┐  ┌─────────────────┐
    │ to_fp16()     │    │ to_int8_syn()│  │to_int8_calibrated
    │ Compress→FP16 │    │Generate syn  │  │Load real data
    │               │    │ data, quant  │  │Calibrate, quant
    └───────┬───────┘    └──────┬───────┘  └────────┬────────┘
            │                   │                   │
    ┌───────▼──────┐    ┌────────▼────┐    ┌────────▼────────┐
    │ fp16/        │    │ int8/       │    │ int8_calibrated/│
    │ model.xml    │    │ model.xml   │    │ model.xml       │
    │ model.bin    │    │ model.bin   │    │ model.bin       │
    └───────┬──────┘    └────────┬────┘    └────────┬────────┘
            │                   │                   │
            └───────────────────┴───────────────────┘
                           │
                           ▼
                 Return ExportResult
                 ├─ model_name
                 ├─ resolution
                 ├─ format
                 ├─ output_path
                 └─ success/error
```

## Deployment Context

```
┌──────────────────────────────────────────────────────────────────┐
│                      Deployment Scenarios                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Scenario 1: GPU Deployment (FP16)                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Developer                                                  │ │
│  │ Export FP16 → models/yolo26n-seg/320/fp16/                │ │
│  │ Package & Deploy on GPU server                            │ │
│  │ Load: ov.Core().read_model("model.xml")                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Scenario 2: CPU Deployment (INT8)                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Step 1: Extract calibration frames                         │ │
│  │ Step 2: Export INT8 → models/yolo26n-seg/320/int8_cal/    │ │
│  │ Step 3: Package & Deploy on CPU server                    │ │
│  │ Load: ov.Core().read_model("model.xml")                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Scenario 3: Edge Deployment (All Formats)                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Export:                                                    │ │
│  │ • ONNX for flexibility                                    │ │
│  │ • FP16 for GPU edge devices                               │ │
│  │ • INT8 for CPU edge devices                               │ │
│  │                                                            │ │
│  │ Device selector at runtime chooses best format            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Dependencies & Integration

```
┌─────────────────────────────────────────────────────────────────┐
│            bakery-exporters Dependencies                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Core (always):                                                │
│  └─ bakery-luna (for catalog/config)                           │
│                                                                  │
│  Optional [onnx]:                                              │
│  └─ ultralytics  ──┬─ PyTorch                                  │
│                    ├─ torchvision                              │
│                    └─ (500MB total)                             │
│                                                                  │
│  Optional [openvino]:                                          │
│  └─ openvino     ──┬─ openvino-dev                             │
│                    ├─ numpy                                    │
│                    └─ (200MB total)                             │
│                                                                  │
│  Optional [nncf]:                                              │
│  └─ nncf         ──┬─ scipy                                    │
│                    ├─ openvino (required)                      │
│                    └─ (100MB total)                             │
│                                                                  │
│  Users install what they need:                                │
│  • pip install bakery-exporters[onnx]                          │
│  • pip install bakery-exporters[openvino]                      │
│  • pip install bakery-exporters[nncf]                          │
│  • pip install bakery-exporters[full]                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Version 0.1.0 Release Components

```
bakery-exporters/0.1.0
├── Core Modules
│   ├── onnx.py          (OnnxExporter)
│   ├── openvino.py      (OpenVINOConverter, OpenVINOPrecision)
│   ├── calibration.py   (CalibrationDataLoader)
│   ├── pipeline.py      (ExportPipeline, ExportFormat, ExportResult)
│   └── cli.py           (CLI utilities)
│
├── Documentation
│   ├── .docs/specs/exporters-specification.md
│   ├── .docs/adrs/001-exporters-architecture.md
│   ├── USAGE.md
│   ├── README.md
│   └── .docs/C4-model.md (this file)
│
├── Configuration
│   ├── pyproject.toml
│   └── Optional dependencies [onnx], [openvino], [nncf], [full]
│
└── Scripts (using exporters)
    ├── ../../scripts/tasks/export/export_sauron_segmentation.py
    ├── ../../scripts/tasks/export/export_sauron_pose.py
    ├── ../../scripts/tasks/export/export_int8.py
    └── ../../scripts/tasks/export/calibrate_int8.py
```
