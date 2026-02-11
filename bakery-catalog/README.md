# Bakery Catalog

This package provides the single source of truth for Bakery model paths and configuration.

## Usage

```python
from bakery_catalog import ModelPath, config

# Get model path
path = ModelPath.get("yolo26n-seg", 320, "fp16")
```
