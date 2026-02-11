"""
Bakery Runtime - Model Inference Abstraction Layer.
====================================================

The "mecánico" (mechanic) — takes model artifacts from the catalog,
compiles them for the target device, and delivers ready-to-use instances.

Usage:
    from bakery_runtime import ModelInstance, Device
    from bakery_runtime.filtering import FilterPolicy

    instance = ModelInstance.from_info(model_info, device=Device.GPU)
    outputs = instance.infer(tensor)
"""

from bakery_runtime.model_instance import ModelInstance, Device
from bakery_runtime.benchmark import BenchmarkRunner, BenchmarkResult, CPUMonitor
from bakery_runtime.filtering import FilterPolicy

__all__ = [
    "ModelInstance",
    "Device",
    "FilterPolicy",
    "BenchmarkRunner",
    "BenchmarkResult",
    "CPUMonitor",
]
