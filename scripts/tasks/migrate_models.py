"""
Bakery - Model Migration Script
===============================

Migrates existing models from legacy 'exports/' directory
to the new standard 'models/' structure.

Legacy Structure (detected):
    exports/fp16/sauron_pose_yolo26/medium/yolo26n-pose_320_fp16_gpu/model.xml

New Structure:
    models/yolo26n-pose/320/fp16/model.xml

Usage:
    uv run migrate_models.py
    uv run migrate_models.py --dry-run
"""

import sys
import shutil
import re
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Use new catalog
from bakery_catalog import config, ModelPath, FormatType
from bakery_catalog.paths import ModelPath as ConfigModelPath

def parse_legacy_dirname(dirname: str) -> dict | None:
    """
    Parse info from directory name like 'yolo26n-pose_320_fp16_gpu'.
    """
    # Pattern: yolo<ver><size>-<task>_<res>_<fmt>_<device>
    # e.g. yolo26n-pose_320_fp16_gpu
    # e.g. yolo26n-seg_320_fp16_gpu (guessing for seg)
    
    # Also handle 'yolo26n-seg_320_int8_openvino' maybe?
    
    # Regex:
    # ^(yolo\d+[nsmlx]-(?:pose|seg|det))_(\d+)_([a-zA-Z0-9_]+)(?:_.*)?$
    
    match = re.match(r"^(yolo\d+[nsmlx]-(?:pose|seg))_(\d+)_([a-z0-9]+)(?:_.*)?$", dirname)
    if not match:
        return None
        
    model_name = match.group(1)
    resolution = int(match.group(2))
    fmt_raw = match.group(3)
    
    # Map format
    fmt = None
    if fmt_raw == "fp16":
        fmt = "fp16"
    elif fmt_raw == "int8":
        fmt = "int8"
    elif fmt_raw == "onnx":
        fmt = "onnx"
    
    # Also check if 'calibrated' is in the name or trailing part?
    # For now assume mostly fp16/int8/onnx match directly or as prefix.
    
    if "calibrated" in dirname:
        fmt = "int8_calibrated"
        
    if not fmt:
        return None
        
    return {
        "model_name": model_name,
        "resolution": resolution,
        "format": fmt
    }

def migrate_models(dry_run: bool = False):
    """
    Scan and migrate models.
    """
    base_dirs = [
        PROJECT_ROOT / "exports",
        PROJECT_ROOT / "models/exports"
    ]
    
    print(f"🎯 Bakery - Model Migration")
    print("=" * 70)
    print(f"Scanning: {[str(d.relative_to(PROJECT_ROOT)) for d in base_dirs if d.exists()]}")
    print(f"Target:   {config.models_dir.absolute()}")
    print("-" * 70)
    
    migrated_count = 0
    
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
            
        # Walk recursively looking for model files
        for path in base_dir.rglob("*"):
            if not path.is_dir():
                continue
                
            # Check if this directory looks like a model leaf
            info = parse_legacy_dirname(path.name)
            if not info:
                continue
                
            # Verify contents
            has_xml = (path / "model.xml").exists()
            has_onnx = (path / "model.onnx").exists()
            
            if not has_xml and not has_onnx:
                continue
                
            # Construct target path
            target_dir = ModelPath.get_dir(
                info["model_name"],
                info["resolution"],
                info["format"]
            )
            
            print(f"\nFound: {path.relative_to(PROJECT_ROOT)}")
            print(f"  -> {target_dir.relative_to(PROJECT_ROOT)}")
            
            if dry_run:
                print("  [DRY RUN] Would copy files")
                continue
                
            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            for file in path.iterdir():
                if file.is_file():
                    shutil.copy2(file, target_dir / file.name)
            
            print("  ✅ Migrated")
            migrated_count += 1
            
    print("-" * 70)
    print(f"Total migrated: {migrated_count}")

def main():
    parser = argparse.ArgumentParser(description="Migrate legacy models")
    parser.add_argument("--dry-run", action="store_true", help="Simulate migration")
    args = parser.parse_args()
    
    migrate_models(args.dry_run)

if __name__ == "__main__":
    main()
