"""
Bakery CLI - Benchmark Command

Measures inference performance: FPS, latency, CPU load, memory.
Thin wrapper around bakery_runtime.benchmark.BenchmarkRunner.
"""
import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table

from bakery_catalog import ModelRepository, config as catalog_config
from bakery_runtime import ModelInstance, Device as RuntimeDevice
from bakery_runtime.benchmark import BenchmarkRunner, BenchmarkResult

console = Console()


def run(
    video: str = typer.Option(..., help="Video file for benchmarking"),
    models_dir: Path = typer.Option(..., help="Directory containing models"),
    max_frames: int = typer.Option(300, "--frames", "-n", help="Max frames to process"),
    device: str = typer.Option("GPU", "--device", "-d", help="Device: GPU, CPU, AUTO"),
    confidence: float = typer.Option(0.25, help="Confidence threshold"),
):
    """
    Benchmark model inference performance.
    """
    console.rule("[bold yellow]⚡ Bakery Benchmark[/bold yellow]")

    # Discover models
    repo = ModelRepository(models_dir)
    all_models = repo.discover()

    if not all_models:
        console.print(f"[red]❌ No models found in {models_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"   Found {len(all_models)} model(s)")
    console.print(f"   Video: {video}")
    console.print(f"   Frames: {max_frames}")
    console.print(f"   Device: {device}\n")

    # Map device string
    device_map = {"GPU": RuntimeDevice.GPU, "CPU": RuntimeDevice.CPU}
    rt_device = device_map.get(device.upper(), RuntimeDevice.GPU)

    runner = BenchmarkRunner()
    results: List[BenchmarkResult] = []

    for info in all_models:
        try:
            console.print(f"   🔬 Benchmarking {info.model_name} @ {info.resolution}px ({info.precision.value})...")
            instance = ModelInstance.from_info(info, device=rt_device, confidence=confidence)
            result = runner.run(instance, video, max_frames=max_frames, device=device)
            results.append(result)
            console.print(f"      → {result.fps:.1f} FPS, {result.latency_p50_ms:.1f}ms p50")
        except Exception as e:
            console.print(f"      [red]❌ Error: {e}[/red]")

    if not results:
        console.print("[red]No benchmarks completed.[/red]")
        raise typer.Exit(1)

    # Results table
    _print_results_table(results)


def _print_results_table(results: List[BenchmarkResult]):
    """Render benchmark results as a Rich table."""
    table = Table(title="Benchmark Results", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("FPS", justify="right", style="green bold")
    table.add_column("Latency p50", justify="right")
    table.add_column("Latency p95", justify="right")
    table.add_column("Latency p99", justify="right")
    table.add_column("CPU %", justify="right")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("CPU-sec", justify="right", style="dim")

    for r in results:
        table.add_row(
            r.model_name,
            f"{r.fps:.1f}",
            f"{r.latency_p50_ms:.1f} ms",
            f"{r.latency_p95_ms:.1f} ms",
            f"{r.latency_p99_ms:.1f} ms",
            f"{r.avg_cpu_percent:.1f}%",
            f"{r.avg_memory_mb:.0f}",
            f"{r.cpu_seconds:.1f}",
        )

    console.print(table)

    # Analysis
    best_fps = max(results, key=lambda x: x.fps)
    best_load = min(results, key=lambda x: x.cpu_seconds)

    console.print(f"\n   🏆 Best FPS: {best_fps.model_name} ({best_fps.fps:.1f} FPS)")
    console.print(f"   ⚡ Lowest CPU load: {best_load.model_name} ({best_load.cpu_seconds:.1f} CPU-sec)")
