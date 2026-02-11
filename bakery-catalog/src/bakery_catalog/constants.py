"""
Bakery Catalog Constants
========================

Model constants for the Bakery pipeline.
"""

from typing import Literal

# Supported model formats
FormatType = Literal["onnx", "fp16", "int8", "int8_calibrated"]

# YOLO versions available
YOLO_VERSIONS = ["8", "11", "26"]

# Model sizes
MODEL_SIZES = ["n", "s", "m", "l", "x"]

# Task types
TASK_TYPES = ["detection", "segmentation", "pose"]

# Standard resolutions
RESOLUTIONS = [160, 192, 224, 256, 288, 320, 640]
