"""
Bakery CLI - Doctor Command

Checks the runtime environment: devices, drivers, dependencies.
"""
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def run():
    """
    Check environment health: devices, drivers, and dependencies.
    """
    console.rule("[bold green]🩺 Bakery Doctor[/bold green]")
    all_ok = True

    # 1. Python
    import sys
    console.print(f"\n[bold]Python[/bold]")
    console.print(f"   Version: {sys.version.split()[0]}")

    # 2. OpenVINO
    console.print(f"\n[bold]OpenVINO Runtime[/bold]")
    try:
        import openvino as ov
        core = ov.Core()
        console.print(f"   Version:  {ov.__version__}")
        
        devices = core.available_devices
        console.print(f"   Devices:  {', '.join(devices)}")

        # Device details
        table = Table(show_lines=False, box=None, padding=(0, 2))
        table.add_column("Device", style="cyan")
        table.add_column("Full Name")
        table.add_column("Status")

        for device in devices:
            try:
                full_name = core.get_property(device, "FULL_DEVICE_NAME")
                status = "[green]✅ Available[/green]"
            except Exception:
                full_name = "—"
                status = "[yellow]⚠️ Limited info[/yellow]"
            table.add_row(device, full_name, status)

        console.print(table)

        # Check CPU capabilities
        if "CPU" in devices:
            try:
                cpu_name = core.get_property("CPU", "FULL_DEVICE_NAME")
                # Check for VNNI/AVX
                import subprocess
                result = subprocess.run(
                    ["lscpu"], capture_output=True, text=True, timeout=5
                )
                flags = result.stdout.lower()
                has_vnni = "vnni" in flags or "avx512_vnni" in flags
                has_avx512 = "avx512" in flags
                has_avx2 = "avx2" in flags

                console.print(f"\n[bold]CPU Capabilities[/bold]")
                console.print(f"   AVX2:           {'✅' if has_avx2 else '❌'}")
                console.print(f"   AVX-512:        {'✅' if has_avx512 else '❌'}")
                console.print(f"   VNNI (INT8):    {'✅' if has_vnni else '❌'}")

                if not has_vnni:
                    console.print("   [yellow]⚠️  INT8 models will run without hardware acceleration[/yellow]")
            except Exception:
                pass

        # Check GPU
        if "GPU" in devices:
            console.print(f"\n[bold]GPU[/bold]")
            try:
                gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
                console.print(f"   Name: {gpu_name}")
                console.print(f"   Status: [green]✅ Available for inference[/green]")
            except Exception:
                console.print(f"   [yellow]⚠️ GPU detected but details unavailable[/yellow]")

    except ImportError:
        console.print("   [red]❌ OpenVINO not installed[/red]")
        console.print("   Install: pip install openvino")
        all_ok = False

    # 3. Key dependencies
    console.print(f"\n[bold]Dependencies[/bold]")
    deps = {
        "opencv-python": "cv2",
        "numpy": "numpy",
        "supervision": "supervision",
        "ultralytics": "ultralytics",
        "typer": "typer",
        "rich": "rich",
        "nncf": "nncf",
    }

    table = Table(show_lines=False, box=None, padding=(0, 2))
    table.add_column("Package", style="cyan")
    table.add_column("Version")
    table.add_column("Status")

    for name, module in deps.items():
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "—")
            table.add_row(name, version, "[green]✅[/green]")
        except ImportError:
            required = name in ["opencv-python", "numpy", "typer", "rich"]
            status = "[red]❌ Missing (required)[/red]" if required else "[dim]— Optional[/dim]"
            table.add_row(name, "—", status)
            if required:
                all_ok = False

    console.print(table)

    # 4. Bakery packages
    console.print(f"\n[bold]Bakery Packages[/bold]")
    bakery_pkgs = ["bakery_catalog", "bakery_runtime", "bakery_exporters"]
    for pkg in bakery_pkgs:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            console.print(f"   {pkg}: [green]✅[/green] {version}")
        except ImportError:
            console.print(f"   {pkg}: [red]❌ Not installed[/red]")
            all_ok = False

    # Verdict
    console.print()
    if all_ok:
        console.print(Panel("[bold green]✅ Environment is healthy![/bold green]", border_style="green"))
    else:
        console.print(Panel("[bold yellow]⚠️  Some issues found. Review above.[/bold yellow]", border_style="yellow"))
