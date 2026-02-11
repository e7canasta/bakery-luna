"""
Bakery CLI - Main Entry Point
"""
import typer
from rich.console import Console
from typing import Optional

from bakery_cli import __version__

# Create main app
app = typer.Typer(
    name="bakery",
    help="Bakery Vision Pipeline CLI 🥯",
    add_completion=False,
    no_args_is_help=True,
)

# Global console
console = Console()

def version_callback(value: bool):
    if value:
        console.print(f"Bakery CLI v{__version__}")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show version and exit.", callback=version_callback, is_eager=True
    ),
):
    """
    Bakery Vision Pipeline - Unified Command Line Interface.
    """
    pass

# Register subcommands
from bakery_cli.commands import run, validate, export, catalog, doctor, benchmark

# Direct commands (plain functions)
app.command(name="run")(run.run)
app.command(name="validate")(validate.run)
app.command(name="doctor")(doctor.run)
app.command(name="benchmark")(benchmark.run)

# Sub-apps (Typer groups with their own subcommands)
app.add_typer(export.app, name="export")
app.add_typer(catalog.app, name="catalog")

if __name__ == "__main__":
    app()
