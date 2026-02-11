"""
Example 3: Batch Export

Export multiple models and sizes in a single batch operation.

Usage:
    python examples/03_batch_export.py
"""

from bakery_exporters import ExportPipeline, ExportFormat
from bakery.catalog import config


def main():
    print("Bakery Exporters - Example 3: Batch Export")
    print("=" * 60)

    # Create pipeline
    pipeline = ExportPipeline()

    # Export configuration
    yolo_version = "26"
    sizes = ["n", "s"]  # Export multiple sizes
    task = "segmentation"
    resolutions = [256, 320]  # Export multiple resolutions
    formats = [ExportFormat.FP16, ExportFormat.INT8]

    print(f"\nBatch Export Configuration:")
    print(f"  YOLO version: {yolo_version}")
    print(f"  Sizes: {', '.join(sizes)}")
    print(f"  Task: {task}")
    print(f"  Resolutions: {', '.join(map(str, resolutions))}")
    print(f"  Formats: {', '.join(f.value for f in formats)}")
    print(f"  Total: {len(sizes) * len(resolutions)} models × {len(formats)} formats")
    print(f"  Output directory: {config.models_dir}")

    # Export batch
    all_results = pipeline.export_batch(
        yolo_version=yolo_version,
        sizes=sizes,
        task=task,
        resolutions=resolutions,
        formats=formats,
        dry_run=False
    )

    # Show results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    successful = sum(1 for r in all_results if r.success)
    print(f"\nTotal: {successful}/{len(all_results)} exports successful")

    # Show summary by model
    models = {}
    for result in all_results:
        key = f"{result.model_name}@{result.resolution}"
        if key not in models:
            models[key] = {"success": 0, "failed": 0}

        if result.success:
            models[key]["success"] += 1
        else:
            models[key]["failed"] += 1

    print("\nBy Model:")
    for model, stats in sorted(models.items()):
        total = stats["success"] + stats["failed"]
        print(f"  {model}: {stats['success']}/{total} ✓")

    return 0 if successful == len(all_results) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
