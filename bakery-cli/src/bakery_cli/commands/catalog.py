"""
Bakery CLI - Catalog Command

Browse and inspect available models in the local model repository.
"""
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

from bakery_catalog import ModelRepository, ModelType, config

console = Console()

app = typer.Typer(help="Browse and inspect the model catalog.")


@app.command(name="list")
def list_models(
    models_dir: Optional[Path] = typer.Option(
        None, "--models-dir", "-d",
        help=f"Models directory (default: {config.models_dir})"
    ),
    task: Optional[str] = typer.Option(
        None, "-t", "--task",
        help="Filter by task: segmentation, pose, detection"
    ),
    version: Optional[str] = typer.Option(
        None, "-v", "--version",
        help="Filter by YOLO version: 8, 11, 26"
    ),
):
    """
    List all available models in the catalog.
    """
    console.rule("[bold cyan]📦 Model Catalog[/bold cyan]")

    base = models_dir or config.models_dir
    repo = ModelRepository(base)

    # Map task string to ModelType
    task_filter = None
    if task:
        _map = {"segmentation": ModelType.SEGMENTATION, "pose": ModelType.POSE, "detection": ModelType.DETECTION}
        task_filter = _map.get(task.lower())
        if not task_filter:
            console.print(f"[red]Unknown task: {task}. Use: segmentation, pose, detection[/red]")
            raise typer.Exit(1)

    models = repo.discover(task=task_filter, yolo_version=version)

    if not models:
        console.print("[yellow]No models found.[/yellow]")
        console.print(f"   Searched: {base}")
        raise typer.Exit(0)

    # Build table
    table = Table(title=f"Models in {base}", show_lines=False)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Name", style="cyan bold")
    table.add_column("Type", style="magenta")
    table.add_column("Res", justify="right")
    table.add_column("Precision", style="green")
    table.add_column("Size")
    table.add_column("YOLO", justify="center")
    table.add_column("Path", style="dim")

    for i, m in enumerate(models, 1):
        table.add_row(
            str(i),
            m.model_name,
            m.model_type.value,
            str(m.resolution),
            m.precision.value,
            m.model_size.value if m.model_size else "—",
            m.yolo_version or "—",
            str(m.model_path.parent.relative_to(base)),
        )

    console.print(table)
    console.print(f"\n   Total: {len(models)} model(s)")


@app.command(name="info")
def model_info(
    name: str = typer.Argument(..., help="Model name (e.g. yolo26n-seg)"),
    resolution: int = typer.Option(640, "-r", "--resolution", help="Resolution"),
    format: str = typer.Option("fp16", "-f", "--format", help="Format: fp16, int8, onnx"),
    models_dir: Optional[Path] = typer.Option(
        None, "--models-dir", "-d",
        help=f"Models directory (default: {config.models_dir})"
    ),
):
    """
    Show detailed info for a specific model.
    """
    base = models_dir or config.models_dir
    repo = ModelRepository(base)

    info = repo.get(name, resolution, format)

    if not info:
        console.print(f"[red]❌ Model not found: {name} @ {resolution}px ({format})[/red]")
        console.print(f"   Run `bakery catalog list` to see available models.")
        raise typer.Exit(1)

    console.rule(f"[bold cyan]📋 {name}[/bold cyan]")
    console.print(f"   Name:       {info.model_name}")
    console.print(f"   Type:       {info.model_type.value}")
    console.print(f"   Resolution: {info.resolution}px")
    console.print(f"   Precision:  {info.precision.value}")
    console.print(f"   Size:       {info.model_size.value if info.model_size else '—'}")
    console.print(f"   YOLO:       {info.yolo_version or '—'}")
    console.print(f"   Path:       {info.model_path}")
    console.print(f"   Exists:     {'✅' if info.exists() else '❌'}")

    # File sizes
    if info.exists():
        file_size_mb = info.model_path.stat().st_size / (1024 * 1024)
        console.print(f"   File size:  {file_size_mb:.1f} MB")

        bin_path = info.model_path.with_suffix(".bin")
        if bin_path.exists():
            bin_size_mb = bin_path.stat().st_size / (1024 * 1024)
            console.print(f"   Weights:    {bin_size_mb:.1f} MB")
            console.print(f"   Total:      {(file_size_mb + bin_size_mb):.1f} MB")
