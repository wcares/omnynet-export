"""
Backend compatibility checking for wonnx and ort.

Validates ONNX models against wonnx operator support and shape requirements.
"""

from dataclasses import dataclass, field
from typing import Optional
import onnx
from onnx import ModelProto


# Operators supported by wonnx (based on wonnx source code)
# This list should be kept in sync with wonnx/src/compiler.rs
WONNX_SUPPORTED_OPS = {
    # Basic math
    "Abs", "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log",
    "Neg", "Reciprocal", "Floor", "Ceil", "Round", "Sign",
    "Min", "Max", "Sum", "Mean", "Mod",

    # Trigonometric
    "Sin", "Cos", "Tan", "Asin", "Acos", "Atan",
    "Sinh", "Cosh", "Tanh", "Asinh", "Acosh", "Atanh",

    # Comparison & Logic
    "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
    "And", "Or", "Xor", "Not",

    # Activations
    "Relu", "LeakyRelu", "PRelu", "Sigmoid", "Tanh", "Softmax",
    "Elu", "Celu", "Selu", "Gelu", "Silu", "Mish",
    "HardSigmoid", "HardSwish", "Softplus", "Softsign",
    "Erf",

    # Normalization
    "BatchNormalization", "LayerNormalization", "InstanceNormalization",

    # Convolution
    "Conv", "ConvTranspose",

    # Pooling
    "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool",

    # Linear
    "Gemm", "MatMul",

    # Tensor manipulation
    "Reshape", "Transpose", "Concat", "Split", "Slice",
    "Squeeze", "Unsqueeze", "Flatten", "Expand",
    "Gather", "GatherElements", "GatherND",
    "Scatter", "ScatterElements", "ScatterND",
    "Pad", "Tile",

    # Reduction
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin",
    "ReduceProd", "ReduceL1", "ReduceL2",
    "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare",

    # Type conversion
    "Cast", "CastLike",

    # Conditional
    "Where",

    # Other
    "Identity", "Dropout", "Constant", "ConstantOfShape",
    "Shape", "Size",
    "Clip", "OneHot",
    "Resize",

    # Fused ops (wonnx-specific optimizations)
    "ConvRelu", "ConvLeakyRelu", "ConvMish",
    "GemmBiasRelu", "FlashAttention",
}

# Operators that ort supports but wonnx doesn't (commonly used ones)
WONNX_UNSUPPORTED_COMMON = {
    "Einsum",           # Used in transformers
    "GridSample",       # Spatial transformers
    "NonMaxSuppression", # Detection models
    "RoiAlign",         # Mask R-CNN
    "TopK",             # Sampling
    "ArgMax", "ArgMin", # Argmax operations
    "GRU", "LSTM", "RNN", # Recurrent layers
    "Attention",        # Some attention implementations
    "Range",            # Index generation
    "Loop", "If", "Scan", # Control flow
    "SequenceInsert", "SequenceAt", # Sequence ops
    "StringNormalizer", # String ops
}


@dataclass
class DynamicDimInfo:
    """Information about a dynamic dimension."""
    tensor_name: str
    dim_index: int
    dim_param: str  # The symbolic name like "batch" or "sequence_length"


@dataclass
class UnsupportedOpInfo:
    """Information about an unsupported operator."""
    op_type: str
    node_names: list[str] = field(default_factory=list)
    count: int = 0


@dataclass
class CompatibilityReport:
    """Report of model compatibility with different backends."""

    # ort compatibility (almost always True)
    ort_compatible: bool = True
    ort_cuda: bool = True
    ort_rocm: bool = True
    ort_openvino: bool = True
    ort_cpu: bool = True

    # wonnx compatibility
    wonnx_compatible: bool = True
    wonnx_unsupported_ops: list[UnsupportedOpInfo] = field(default_factory=list)

    # Shape issues
    has_dynamic_shapes: bool = False
    dynamic_dims: list[DynamicDimInfo] = field(default_factory=list)
    shapes_fixed: bool = False

    # Opset
    opset_version: int = 0
    opset_compatible: bool = True

    # Summary
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def is_fully_compatible(self) -> bool:
        """Check if model works on all backends."""
        return self.ort_compatible and self.wonnx_compatible and not self.has_dynamic_shapes

    def get_compatible_backends(self) -> list[str]:
        """Get list of compatible backends."""
        backends = []
        if self.ort_cuda:
            backends.append("ort-CUDA")
        if self.ort_rocm:
            backends.append("ort-ROCm")
        if self.ort_openvino:
            backends.append("ort-OpenVINO")
        if self.ort_cpu:
            backends.append("ort-CPU")
        if self.wonnx_compatible and not self.has_dynamic_shapes:
            backends.append("wonnx-Vulkan")
            backends.append("wonnx-Metal")
            backends.append("wonnx-DX12")
        return backends

    def get_incompatible_backends(self) -> list[str]:
        """Get list of incompatible backends with reasons."""
        incompatible = []
        if not self.wonnx_compatible:
            ops = ", ".join(op.op_type for op in self.wonnx_unsupported_ops[:3])
            if len(self.wonnx_unsupported_ops) > 3:
                ops += f", ... (+{len(self.wonnx_unsupported_ops) - 3} more)"
            incompatible.append(f"wonnx (unsupported ops: {ops})")
        elif self.has_dynamic_shapes and not self.shapes_fixed:
            dims = ", ".join(d.dim_param for d in self.dynamic_dims[:3])
            incompatible.append(f"wonnx (dynamic shapes: {dims})")
        return incompatible


def check_compatibility(model: ModelProto) -> CompatibilityReport:
    """
    Check model compatibility with ort and wonnx backends.

    Args:
        model: ONNX model to check

    Returns:
        CompatibilityReport with detailed compatibility information
    """
    report = CompatibilityReport()

    # Get opset version
    for oi in model.opset_import:
        if oi.domain == "" or oi.domain == "ai.onnx":
            report.opset_version = oi.version
            break

    # Check opset compatibility (wonnx works best with opset 12-17)
    if report.opset_version < 12:
        report.warnings.append(f"Opset {report.opset_version} is old, consider upgrading to 17")

    # Check operators
    op_usage: dict[str, list[str]] = {}  # op_type -> list of node names

    for node in model.graph.node:
        if node.op_type not in op_usage:
            op_usage[node.op_type] = []
        op_usage[node.op_type].append(node.name or f"unnamed_{len(op_usage[node.op_type])}")

    # Find unsupported ops for wonnx
    for op_type, node_names in op_usage.items():
        if op_type not in WONNX_SUPPORTED_OPS:
            report.wonnx_unsupported_ops.append(UnsupportedOpInfo(
                op_type=op_type,
                node_names=node_names[:5],  # Limit to first 5
                count=len(node_names),
            ))

    if report.wonnx_unsupported_ops:
        report.wonnx_compatible = False
        report.errors.append(
            f"{len(report.wonnx_unsupported_ops)} operator(s) not supported by wonnx"
        )

    # Check for dynamic shapes in inputs
    for inp in model.graph.input:
        # Skip initializers
        if any(init.name == inp.name for init in model.graph.initializer):
            continue

        if inp.type.HasField("tensor_type"):
            tt = inp.type.tensor_type
            if tt.HasField("shape"):
                for i, dim in enumerate(tt.shape.dim):
                    if not dim.HasField("dim_value"):
                        # Dynamic dimension
                        dim_param = dim.dim_param if dim.HasField("dim_param") else f"dim_{i}"
                        report.dynamic_dims.append(DynamicDimInfo(
                            tensor_name=inp.name,
                            dim_index=i,
                            dim_param=dim_param,
                        ))

    if report.dynamic_dims:
        report.has_dynamic_shapes = True
        report.warnings.append(
            f"{len(report.dynamic_dims)} dynamic dimension(s) found - wonnx requires fixed shapes"
        )

    # ort is compatible with almost everything
    # Only mark incompatible for extremely unusual cases
    report.ort_compatible = True

    return report


def fix_dynamic_shapes(
    model: ModelProto,
    input_shapes: dict[str, list[int]],
) -> tuple[ModelProto, list[str]]:
    """
    Fix dynamic shapes in model by setting concrete dimensions.

    Args:
        model: ONNX model with dynamic shapes
        input_shapes: Dict mapping input names to concrete shapes

    Returns:
        Tuple of (fixed model, list of changes made)
    """
    import copy
    model = copy.deepcopy(model)
    changes = []

    # Fix input shapes
    for inp in model.graph.input:
        if inp.name in input_shapes:
            shape = input_shapes[inp.name]
            if inp.type.HasField("tensor_type"):
                tt = inp.type.tensor_type
                if tt.HasField("shape"):
                    for i, (dim, new_val) in enumerate(zip(tt.shape.dim, shape)):
                        if not dim.HasField("dim_value") or dim.dim_value != new_val:
                            old_val = dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                            dim.Clear()
                            dim.dim_value = new_val
                            changes.append(f"{inp.name}[{i}]: {old_val} -> {new_val}")

    # Run shape inference to propagate
    try:
        model = onnx.shape_inference.infer_shapes(model)
        changes.append("Shape inference completed")
    except Exception as e:
        changes.append(f"Shape inference warning: {e}")

    return model, changes


def get_wonnx_supported_ops() -> set[str]:
    """Get the set of operators supported by wonnx."""
    return WONNX_SUPPORTED_OPS.copy()


def format_compatibility_report(report: CompatibilityReport, use_color: bool = True) -> str:
    """
    Format compatibility report as a string for display.

    Args:
        report: The compatibility report
        use_color: Whether to use ANSI colors

    Returns:
        Formatted string
    """
    if use_color:
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
    else:
        GREEN = RED = YELLOW = BOLD = RESET = ""

    lines = []
    lines.append("")
    lines.append(f"{BOLD}{'=' * 60}{RESET}")
    lines.append(f"{BOLD}           Backend Compatibility Report{RESET}")
    lines.append(f"{BOLD}{'=' * 60}{RESET}")
    lines.append("")

    # ort backends
    lines.append(f"  {BOLD}ort backends:{RESET}")
    lines.append(f"    CUDA (NVIDIA)      {GREEN}[OK]{RESET} Full support")
    lines.append(f"    ROCm (AMD)         {GREEN}[OK]{RESET} Full support")
    lines.append(f"    OpenVINO (Intel)   {GREEN}[OK]{RESET} Full support")
    lines.append(f"    CPU                {GREEN}[OK]{RESET} Full support")
    lines.append("")

    # wonnx backends
    lines.append(f"  {BOLD}wonnx backends:{RESET}")

    if report.wonnx_compatible and not report.has_dynamic_shapes:
        lines.append(f"    Vulkan (Linux/Win) {GREEN}[OK]{RESET} Full support")
        lines.append(f"    Metal (macOS)      {GREEN}[OK]{RESET} Full support")
        lines.append(f"    DX12 (Windows)     {GREEN}[OK]{RESET} Full support")
    elif not report.wonnx_compatible:
        n_ops = len(report.wonnx_unsupported_ops)
        lines.append(f"    Vulkan (Linux/Win) {RED}[FAIL]{RESET} {n_ops} unsupported op(s)")
        lines.append(f"    Metal (macOS)      {RED}[FAIL]{RESET} {n_ops} unsupported op(s)")
        lines.append(f"    DX12 (Windows)     {RED}[FAIL]{RESET} {n_ops} unsupported op(s)")
    elif report.has_dynamic_shapes and not report.shapes_fixed:
        n_dims = len(report.dynamic_dims)
        lines.append(f"    Vulkan (Linux/Win) {YELLOW}[WARN]{RESET} {n_dims} dynamic dim(s)")
        lines.append(f"    Metal (macOS)      {YELLOW}[WARN]{RESET} {n_dims} dynamic dim(s)")
        lines.append(f"    DX12 (Windows)     {YELLOW}[WARN]{RESET} {n_dims} dynamic dim(s)")

    lines.append("")

    # Unsupported ops details
    if report.wonnx_unsupported_ops:
        lines.append(f"  {BOLD}Unsupported operators for wonnx:{RESET}")
        for op_info in report.wonnx_unsupported_ops[:10]:
            nodes_str = ", ".join(op_info.node_names[:3])
            if op_info.count > 3:
                nodes_str += f" (+{op_info.count - 3} more)"
            lines.append(f"    {RED}*{RESET} {op_info.op_type} (nodes: {nodes_str})")
        if len(report.wonnx_unsupported_ops) > 10:
            lines.append(f"    ... and {len(report.wonnx_unsupported_ops) - 10} more")
        lines.append("")

    # Dynamic shapes details
    if report.has_dynamic_shapes and not report.shapes_fixed:
        lines.append(f"  {BOLD}Dynamic dimensions (need fixing for wonnx):{RESET}")
        seen = set()
        for dim_info in report.dynamic_dims:
            key = f"{dim_info.tensor_name}[{dim_info.dim_index}]"
            if key not in seen:
                seen.add(key)
                lines.append(f"    {YELLOW}*{RESET} {key}: {dim_info.dim_param}")
        lines.append("")
        lines.append(f"  {YELLOW}Hint:{RESET} Use --shape \"{report.dynamic_dims[0].tensor_name}:1,3,224,224\" to fix")
        lines.append("")

    lines.append(f"{'=' * 60}")

    # Summary
    if report.is_fully_compatible():
        lines.append(f"{GREEN}Model compatible with ALL backends.{RESET}")
    elif report.wonnx_compatible and report.has_dynamic_shapes:
        lines.append(f"{YELLOW}WARNING: Model needs shape fixing for wonnx backend.{RESET}")
        lines.append(f"         Systems without CUDA/ROCm/OpenVINO will use CPU fallback.")
    else:
        lines.append(f"{RED}WARNING: Model will FAIL on wonnx backend.{RESET}")
        lines.append(f"         Systems without CUDA/ROCm/OpenVINO will use CPU fallback.")
        lines.append("")
        lines.append(f"  Affected systems: Intel iGPU, older AMD GPUs, ARM devices")
        lines.append("")
        lines.append(f"  Options:")
        lines.append(f"    1. Add missing ops to wonnx (github.com/wcares/wonnx)")
        lines.append(f"    2. Replace ops in model with supported alternatives")
        lines.append(f"    3. Accept CPU fallback on affected systems")

    lines.append("")

    return "\n".join(lines)
