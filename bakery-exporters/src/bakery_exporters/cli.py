"""
Bakery Exporters - CLI
======================

Command-line interface for model export.

Consolidates the ~80 lines of argument parsing duplicated across:
- export_sauron_segmentation.py
- export_sauron_pose.py
- export_int8.py
- calibrate_int8.py
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import List

from bakery.catalog import (
    config,
    YOLO_VERSIONS,
    MODEL_SIZES,
    TASK_TYPES,
    RESOLUTIONS,
)

from bakery_exporters.pipeline import ExportPipeline, ExportFormat


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """
    Create base argument parser with common arguments.

    Args:
        description: Parser description

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=YOLO_VERSIONS,
        default=config.default_yolo_version,
        help=f"YOLO version (default: {config.default_yolo_version})"
    )

    parser.add_argument(
        "--model-size", "-s",
        type=str,
        nargs="+",
        choices=MODEL_SIZES,
        help="Model size(s)"
    )

    parser.add_argument(
        "--type", "-t",
        type=str,
        choices=TASK_TYPES,
        default="detection",
        help="Task type (default: detection)"
    )

    parser.add_argument(
        "--resolution", "-r",
        type=int,
        nargs="+",
        choices=RESOLUTIONS,
        help="Resolution(s)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all combinations"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without running"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help=f"Override models directory (default: {config.models_dir})"
    )

    return parser


def add_format_argument(
    parser: argparse.ArgumentParser,
    choices: List[str],
    default: List[str],
) -> None:
    """
    Add format argument to parser.

    Args:
        parser: ArgumentParser to modify
        choices: Valid format choices
        default: Default format(s)
    """
    parser.add_argument(
        "--format", "-f",
        type=str,
        nargs="+",
        choices=choices,
        default=default,
        help=f"Output format(s) (default: {default})"
    )


def main() -> int:
    """
    Main CLI entry point for bakery-export command.

    Returns:
        Exit code (0 for success)
    """
    parser = create_base_parser(
        "Export YOLO models to various formats\n\n"
        "Examples:\n"
        "  bakery-export -m 26 -s n -t segmentation -r 320 -f fp16\n"
        "  bakery-export -m 26 -s n s m -t pose -r 256 320 -f int8\n"
        "  bakery-export -m 26 --all -t segmentation -f fp16"
    )

    add_format_argument(
        parser,
        ["onnx", "fp16", "int8", "int8_calibrated"],
        ["fp16"]
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and (not args.model_size or not args.resolution):
        print("Error: Specify --model-size and --resolution, or use --all")
        parser.print_help()
        return 1

    # Override config if needed
    if args.output:
        config.models_dir = Path(args.output)

    # Determine what to export
    if args.all:
        model_sizes = MODEL_SIZES
        resolutions = RESOLUTIONS
    else:
        model_sizes = args.model_size
        resolutions = args.resolution

    # Convert format strings to enums
    formats = [ExportFormat(f) for f in args.format]

    print("Bakery Exporters - Export Pipeline")
    print("=" * 70)
    print(f"   Directory: {config.models_dir.absolute()}")

    # Run pipeline
    pipeline = ExportPipeline()
    all_results = pipeline.export_batch(
        args.model,
        model_sizes,
        args.type,
        resolutions,
        formats,
        args.dry_run
    )

    # Return success if any exports succeeded
    return 0 if any(r.success for r in all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
