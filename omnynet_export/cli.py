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


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0")
@click.argument("model_path", type=click.Path(exists=True), required=False)
@click.option("--output", "-o", help="Output .omny file path (default: same name as input)")
@click.pass_context
def main(ctx, model_path: Optional[str], output: Optional[str]):
    """OmnyNet Export - Convert models to .omny format for distributed inference.

    Simple usage (auto mode):
        omnynet-export model.onnx          # → model.omny
        omnynet-export model.pt            # → model.omny
        omnynet-export model.onnx -o out.omny

    For more control, use subcommands:
        omnynet-export export ...
        omnynet-export enrich ...
        omnynet-export inspect ...
    """
    # If a model path is provided without subcommand, run auto mode
    if model_path and ctx.invoked_subcommand is None:
        _run_auto_mode(model_path, output)
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def _run_auto_mode(model_path: str, output: Optional[str]):
    """Auto mode: detect model type, convert, validate, done."""
    import numpy as np

    model_path = Path(model_path)

    # Auto-generate output path if not provided
    if output:
        output_path = Path(output)
    else:
        output_path = model_path.with_suffix(".omny")

    # Detect model type
    suffix = model_path.suffix.lower()

    console.print(f"[bold blue]Auto Mode[/]")
    console.print(f"  Input:  {model_path}")
    console.print(f"  Output: {output_path}")
    console.print()

    config = ExportConfig(
        model_name=model_path.stem,
        validate=True,  # Always validate in auto mode
    )

    if suffix == ".onnx":
        console.print("[cyan]Detected:[/] ONNX model → enriching with metadata")
        console.print()

        # Generate sample input from model
        import onnx
        model = onnx.load(str(model_path))
        sample_input = {}
        for inp in model.graph.input:
            if any(init.name == inp.name for init in model.graph.initializer):
                continue
            if inp.type.HasField("tensor_type"):
                tt = inp.type.tensor_type
                shape = []
                for dim in tt.shape.dim:
                    if dim.HasField("dim_value") and dim.dim_value > 0:
                        shape.append(dim.dim_value)
                    else:
                        shape.append(1)
                sample_input[inp.name] = np.random.randn(*shape).astype(np.float32)

        result = enrich_onnx(model_path, output_path, sample_input=sample_input, config=config)

    elif suffix in (".pt", ".pth"):
        console.print("[cyan]Detected:[/] PyTorch model → exporting to ONNX + metadata")
        console.print("[yellow]Note:[/] PyTorch export requires model architecture. Using default input shape.")
        console.print()

        # For PyTorch, we need sample input - use common image shape
        sample_input = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
        result = export_model(model_path, output_path, sample_input, config)

    else:
        console.print(f"[bold red]Error:[/] Unknown file type: {suffix}")
        console.print("Supported: .onnx, .pt, .pth")
        sys.exit(1)

    # Print result
    if result.success:
        console.print(f"[bold green]Success![/] Saved to {output_path}")
        if result.validated:
            console.print(f"[bold green]Cut points validated![/]")
        console.print()
        _print_export_result(result)
    else:
        console.print(f"[bold red]Failed:[/] {result.error}")
        sys.exit(1)


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
@click.option("--sample-input", "-i", type=click.Path(exists=True), help="Sample input .npy file for validation")
@click.option("--validate", "-v", is_flag=True, help="Validate cut points by running actual inference")
@click.option("--min-vram", default=1500, help="Minimum VRAM per node (MB)")
@click.option("--max-shard-size", default=1200, help="Maximum shard size (MB)")
@click.option("--name", help="Model name (default: filename)")
def enrich(
    onnx_path: str,
    output: str,
    sample_input: Optional[str],
    validate: bool,
    min_vram: int,
    max_shard_size: int,
    name: Optional[str],
):
    """Enrich an existing ONNX model with OmnyNet metadata."""
    import numpy as np

    onnx_path = Path(onnx_path)
    output_path = Path(output)

    console.print(f"[bold blue]Enriching:[/] {onnx_path}")

    # Load sample input if provided
    input_data = None
    if sample_input:
        input_data = {"input": np.load(sample_input)}
    elif validate:
        # Generate random input based on model's input shape
        console.print("[yellow]No sample input provided. Generating random input for validation...[/]")
        import onnx as onnx_lib
        model = onnx_lib.load(str(onnx_path))
        input_data = {}
        for inp in model.graph.input:
            # Skip initializers
            if any(init.name == inp.name for init in model.graph.initializer):
                continue
            if inp.type.HasField("tensor_type"):
                tt = inp.type.tensor_type
                shape = []
                for dim in tt.shape.dim:
                    if dim.HasField("dim_value") and dim.dim_value > 0:
                        shape.append(dim.dim_value)
                    else:
                        shape.append(1)  # Default dynamic dim to 1
                input_data[inp.name] = np.random.randn(*shape).astype(np.float32)

    config = ExportConfig(
        min_vram_mb=min_vram,
        max_shard_size_mb=max_shard_size,
        model_name=name,
        validate=validate,
    )

    result = enrich_onnx(onnx_path, output_path, sample_input=input_data, config=config)

    if result.success:
        console.print(f"[bold green]Success![/] Saved to {output_path}")
        if result.validated:
            console.print(f"[bold green]Cut points validated![/]")
        console.print()
        _print_export_result(result)
        if result.validation_errors:
            console.print("\n[yellow]Validation warnings:[/]")
            for err in result.validation_errors:
                console.print(f"  - {err}")
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
    validated_status = "[green]Yes[/]" if meta.export_info.validated else "[yellow]No[/]"
    table.add_row("Cuts Validated", validated_status)

    console.print(table)


if __name__ == "__main__":
    main()
