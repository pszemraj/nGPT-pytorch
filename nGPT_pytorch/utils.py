from typing import List, Optional, Tuple
import torch
import logging
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def check_ampere_gpu():
    """
    Check if the GPU supports NVIDIA Ampere or later and enable FP32 in PyTorch if it does.
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.info("No GPU detected, running on CPU.")
        return

    try:
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        major, minor = capability

        # Check if the GPU is Ampere or newer
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            gpu_name = torch.cuda.get_device_name(device)
            print(
                f"{gpu_name} (compute capability {major}.{minor}) supports NVIDIA Ampere or later, enabled TF32 in PyTorch."
            )
        else:
            gpu_name = torch.cuda.get_device_name(device)
            print(
                f"{gpu_name} (compute capability {major}.{minor}) does not support NVIDIA Ampere or later."
            )

    except Exception as e:
        logging.warning(f"Error occurred while checking GPU: {e}")


def model_summary(
    model: nn.Module, max_depth: int = 4, show_input_size: bool = False
) -> None:
    """
    Prints an accurate summary of the model, avoiding double-counting of parameters.

    :param PreTrainedModel model: torch model to summarize
    :param int max_depth: maximum depth of the model to print, defaults to 4
    :param bool show_input_size: whether to show input size for each layer, defaults to False
    """

    def format_params(num_params: int) -> str:
        return f"{num_params:,}" if num_params > 0 else "--"

    def format_size(size: Optional[List[int]]) -> str:
        return "x".join(str(x) for x in size) if size else "N/A"

    def count_parameters(module: nn.Module) -> Tuple[int, int]:
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
        return total_params, trainable_params

    def recursive_summarize(
        module: nn.Module, depth: int, idx: List[int], prefix: str = ""
    ) -> List[Tuple[str, int, int, int, Optional[List[int]], nn.Module]]:
        summary = []

        total_params, trainable_params = count_parameters(module)

        if depth <= max_depth:
            layer_name = f"{prefix}{type(module).__name__}"
            layer_index = ".".join(map(str, idx))
            param_shape = next(
                (p.shape for p in module.parameters(recurse=False) if p.requires_grad),
                None,
            )
            summary.append(
                (layer_name, depth, total_params, trainable_params, param_shape, module)
            )

            for i, (name, child) in enumerate(module.named_children(), 1):
                child_summary = recursive_summarize(
                    child, depth + 1, idx + [i], prefix + "  "
                )
                summary.extend(child_summary)

        return summary

    summary = recursive_summarize(model, 1, [1])

    max_name_length = max(len(name) for name, _, _, _, _, _ in summary)
    max_shape_length = max(len(format_size(shape)) for _, _, _, _, shape, _ in summary)

    print("=" * (max_name_length + 50))
    header = f"{'Layer (type:depth-idx)':<{max_name_length}} {'Output Shape':>{max_shape_length}} {'Param #':>12} {'Trainable':>10}"
    print(header)
    print("=" * (max_name_length + 50))

    for name, depth, num_params, trainable_params, shape, _ in summary:
        shape_str = format_size(shape) if show_input_size else ""
        print(
            f"{name:<{max_name_length}} {shape_str:>{max_shape_length}} {format_params(num_params):>12} {str(trainable_params > 0):>10}"
        )

    total_params, trainable_params = count_parameters(model)
    print("=" * (max_name_length + 50))
    print(f"Total params: {format_params(total_params)}")
    print(f"Trainable params: {format_params(trainable_params)}")
    print(f"Non-trainable params: {format_params(total_params - trainable_params)}")
    print("=" * (max_name_length + 50))
