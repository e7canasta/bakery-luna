"""
Bakery CLI - Export Command

Thin wrapper around bakery_exporters.ExportPipeline.
"""
import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table

from bakery_exporters.pipeline import ExportPipeline, ExportFormat, ExportResult
from bakery_catalog import YOLO_VERSIONS, MODEL_SIZES, TASK_TYPES, RESOLUTIONS, config

console = Console()

app = typer.Typer(help="Export models to various formats (ONNX, FP16, INT8).")


@app.callback(invoke_without_command=True)
def export(
    ctx: typer.Context,
    model: str = typer.Option(
        config.default_yolo_version, "-m", "--model",
        help=f"YOLO version ({', '.join(YOLO_VERSIONS)})"
    ),
    size: List[str] = typer.Option(
        ["n"], "-s", "--size",
        help=f"Model size(s) ({', '.join(MODEL_SIZES)})"
    ),
    task: str = typer.Option(
        "segmentation", "-t", "--task",
        help=f"Task type ({', '.join(TASK_TYPES)})"
    ),
    resolution: List[int] = typer.Option(
        [640], "-r", "--resolution",
        help=f"Resolution(s) ({', '.join(map(str, RESOLUTIONS))})"
    ),
    format: List[str] = typer.Option(
        ["fp16"], "-f", "--format",
        help="Output format(s): onnx, fp16, int8, int8_calibrated"
    ),
    calibrate: bool = typer.Option(
        False, "--calibrate",
        help="Include INT8 calibrated quantization (requires calibration data)"
    ),
    all_combos: bool = typer.Option(
        False, "--all",
        help="Export all size/resolution combinations"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "-o", "--output",
        help=f"Override models directory (default: {config.models_dir})"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show what would be exported without running"
    ),
):
    """
    Export YOLO models to ONNX, FP16, or INT8 formats.

    Examples:
        bakery export -m 26 -s n -t segmentation -r 320 -f fp16
        bakery export -m 26 -s n s -t pose -r 320 640 -f fp16 int8
        bakery export --all -t segmentation -f fp16 --calibrate
    """
    # If a subcommand was invoked, skip
    if ctx.invoked_subcommand is not None:
        return

    console.rule("[bold magenta]🥯 Bakery Export Pipeline[/bold magenta]")

    # Override output dir
    if output_dir:
        config.models_dir = output_dir

    # Resolve sizes/resolutions
    if all_combos:
        sizes = list(MODEL_SIZES)
        resolutions = list(RESOLUTIONS)
    else:
        sizes = size
        resolutions = resolution

    # Build format list
    formats = [ExportFormat(f) for f in format]
    if calibrate and ExportFormat.INT8_CALIBRATED not in formats:
        formats.append(ExportFormat.INT8_CALIBRATED)

    # Summary
    console.print(f"   Model: YOLO {model}")
    console.print(f"   Sizes: {', '.join(sizes)}")
    console.print(f"   Task: {task}")
    console.print(f"   Resolutions: {', '.join(map(str, resolutions))}")
    console.print(f"   Formats: {', '.join(f.value for f in formats)}")
    if dry_run:
        console.print("   [yellow]DRY RUN — no files will be created[/yellow]")

    # Run pipeline
    pipeline = ExportPipeline()
    all_results = pipeline.export_batch(
        yolo_version=model,
        sizes=sizes,
        task=task,
        resolutions=resolutions,
        formats=formats,
        dry_run=dry_run,
    )

    # Results table
    _print_results_table(all_results)


def _print_results_table(results: List[ExportResult]):
    """Render export results as a Rich table."""
    if not results:
        console.print("[yellow]No exports performed.[/yellow]")
        return

    table = Table(title="Export Results", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Resolution", justify="right")
    table.add_column("Format", style="magenta")
    table.add_column("Status")
    table.add_column("Path")

    for r in results:
        status = "[green]✅ OK[/green]" if r.success else f"[red]❌ {r.error}[/red]"
        table.add_row(
            r.model_name,
            str(r.resolution),
            r.format.value,
            status,
            str(r.output_path) if r.success else "—",
        )

    console.print(table)

    ok = sum(1 for r in results if r.success)
    console.print(f"\n✨ {ok}/{len(results)} exports completed successfully.")
