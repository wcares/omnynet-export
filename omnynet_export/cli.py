"""
Command-line interface for omnynet-export.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .exporter import export_model, enrich_onnx, ExportConfig
from .format import inspect_omny, validate_omny

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """OmnyNet Export - Convert models to .omny format for distributed inference."""
    pass


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output .omny file path")
@click.option("--sample-input", "-i", type=click.Path(exists=True), help="Sample input .npy file")
@click.option("--min-vram", default=1500, help="Minimum VRAM per node (MB)")
@click.option("--max-shard-size", default=1200, help="Maximum shard size (MB)")
@click.option("--name", help="Model name (default: filename)")
def export(
    model_path: str,
    output: str,
    sample_input: Optional[str],
    min_vram: int,
    max_shard_size: int,
    name: Optional[str],
):
    """Export a PyTorch model to .omny format."""
    import numpy as np

    model_path = Path(model_path)
    output_path = Path(output)

    console.print(f"[bold blue]Exporting:[/] {model_path}")

    # Load sample input if provided
    if sample_input:
        input_data = {"input": np.load(sample_input)}
    else:
        console.print("[yellow]Warning:[/] No sample input provided. Using default shape.")
        input_data = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}

    config = ExportConfig(
        min_vram_mb=min_vram,
        max_shard_size_mb=max_shard_size,
        model_name=name,
    )

    # Check if it's an ONNX file or PyTorch
    if model_path.suffix in (".onnx",):
        result = enrich_onnx(model_path, output_path, input_data, config)
    else:
        result = export_model(model_path, output_path, input_data, config)

    if result.success:
        console.print(f"[bold green]Success![/] Saved to {output_path}")
        console.print()
        _print_export_result(result)
    else:
        console.print(f"[bold red]Failed:[/] {result.error}")
        sys.exit(1)


@main.command()
@click.argument("onnx_path", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output .omny file path")
@click.option("--min-vram", default=1500, help="Minimum VRAM per node (MB)")
@click.option("--max-shard-size", default=1200, help="Maximum shard size (MB)")
@click.option("--name", help="Model name (default: filename)")
def enrich(
    onnx_path: str,
    output: str,
    min_vram: int,
    max_shard_size: int,
    name: Optional[str],
):
    """Enrich an existing ONNX model with OmnyNet metadata."""
    onnx_path = Path(onnx_path)
    output_path = Path(output)

    console.print(f"[bold blue]Enriching:[/] {onnx_path}")

    config = ExportConfig(
        min_vram_mb=min_vram,
        max_shard_size_mb=max_shard_size,
        model_name=name,
    )

    result = enrich_onnx(onnx_path, output_path, config=config)

    if result.success:
        console.print(f"[bold green]Success![/] Saved to {output_path}")
        console.print()
        _print_export_result(result)
    else:
        console.print(f"[bold red]Failed:[/] {result.error}")
        sys.exit(1)


@main.command()
@click.argument("omny_path", type=click.Path(exists=True))
def inspect(omny_path: str):
    """Inspect an .omny file and show its metadata."""
    omny_path = Path(omny_path)

    info = inspect_omny(omny_path)

    if not info.is_valid:
        console.print(f"[bold red]Invalid .omny file:[/] {info.error}")
        sys.exit(1)

    console.print(Panel(f"[bold]{omny_path.name}[/]", title="OmnyNet Model"))

    if info.metadata:
        _print_metadata(info.metadata)


@main.command()
@click.argument("omny_path", type=click.Path(exists=True))
def validate(omny_path: str):
    """Validate an .omny file."""
    omny_path = Path(omny_path)

    is_valid, errors = validate_omny(omny_path)

    if is_valid:
        console.print(f"[bold green]Valid![/] {omny_path}")
    else:
        console.print(f"[bold red]Invalid:[/] {omny_path}")
        for error in errors:
            console.print(f"  - {error}")
        sys.exit(1)


def _print_export_result(result):
    """Print export result in a nice format."""
    if not result.metadata:
        return

    meta = result.metadata

    # Model info
    table = Table(title="Model Info", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Name", meta.model.name)
    table.add_row("Parameters", f"{meta.model.total_params:,}")
    table.add_row("Size", f"{meta.model.total_size_mb} MB")
    table.add_row("Inference Memory", f"{meta.model.inference_memory_mb} MB")

    console.print(table)
    console.print()

    # Cut points
    if result.cut_points:
        table = Table(title="Cut Points")
        table.add_column("ID", style="cyan")
        table.add_column("After Node")
        table.add_column("Shape")
        table.add_column("Memory (MB)")

        for cp in result.cut_points:
            table.add_row(
                cp.id,
                cp.after_node[:30] + "..." if len(cp.after_node) > 30 else cp.after_node,
                str(cp.shape),
                str(cp.shard_memory_mb),
            )

        console.print(table)
        console.print()

    # Sharding
    console.print(f"[bold]Allowed Shards:[/] {result.allowed_shards}")
    console.print(f"[bold]Min VRAM:[/] {meta.sharding.min_vram_mb} MB")
    console.print(f"[bold]Max Shard Size:[/] {meta.sharding.max_shard_size_mb} MB")


def _print_metadata(meta):
    """Print metadata in a nice format."""
    # Model info
    table = Table(title="Model Info", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Name", meta.model.name)
    table.add_row("Architecture", meta.model.architecture)
    table.add_row("Parameters", f"{meta.model.total_params:,}")
    table.add_row("Size", f"{meta.model.total_size_mb} MB")
    table.add_row("Inference Memory", f"{meta.model.inference_memory_mb} MB")

    console.print(table)
    console.print()

    # Inputs
    if meta.inputs:
        table = Table(title="Inputs")
        table.add_column("Name", style="cyan")
        table.add_column("Shape")
        table.add_column("Dtype")

        for inp in meta.inputs:
            table.add_row(inp.name, str(inp.shape), inp.dtype)

        console.print(table)
        console.print()

    # Outputs
    if meta.outputs:
        table = Table(title="Outputs")
        table.add_column("Name", style="cyan")
        table.add_column("Shape")
        table.add_column("Dtype")

        for out in meta.outputs:
            table.add_row(out.name, str(out.shape), out.dtype)

        console.print(table)
        console.print()

    # Cut points
    if meta.cut_points:
        table = Table(title="Cut Points")
        table.add_column("ID", style="cyan")
        table.add_column("After Node")
        table.add_column("Shape")
        table.add_column("Cumulative (MB)")
        table.add_column("Shard (MB)")

        for cp in meta.cut_points:
            table.add_row(
                cp.id,
                cp.after_node[:25] + "..." if len(cp.after_node) > 25 else cp.after_node,
                str(cp.shape),
                str(cp.cumulative_memory_mb),
                str(cp.shard_memory_mb),
            )

        console.print(table)
        console.print()

    # Sharding
    table = Table(title="Sharding Configuration", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Min VRAM", f"{meta.sharding.min_vram_mb} MB")
    table.add_row("Max Shard Size", f"{meta.sharding.max_shard_size_mb} MB")
    table.add_row("Allowed Shards", str(meta.sharding.allowed_shards))

    console.print(table)
    console.print()

    # Export info
    table = Table(title="Export Info", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Exported At", meta.export_info.exported_at)
    table.add_row("Exporter Version", meta.export_info.exporter_version)
    table.add_row("Source Framework", meta.export_info.source_framework)
    table.add_row("ONNX Opset", str(meta.export_info.onnx_opset))

    console.print(table)


if __name__ == "__main__":
    main()
