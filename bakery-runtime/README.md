# Bakery Runtime

Model inference abstraction layer — the "mecánico" (mechanic).

Receives model artifacts from `bakery-catalog`, compiles them for the target device, and returns opaque `ModelInstance` objects ready for inference.

## Usage

```python
from bakery_catalog import ModelRepository
from bakery_runtime import ModelInstance

# Get model from catalog
repo = ModelRepository()
info = repo.get("yolo26n-seg", 320, "fp16")

# Create a ready-to-use instance
instance = ModelInstance.from_info(info, device="GPU", confidence=0.25)

# Infer — consumer doesn't know the engine underneath
outputs = instance.infer(tensor)
```

## Architecture

- `ModelInstance`: Opaque inference wrapper. Consumer only sees `.infer()`.
- `engines/openvino_engine.py`: OpenVINO-specific compilation and execution.
- Future: `engines/onnxruntime_engine.py`, `engines/torch_engine.py`, etc.
