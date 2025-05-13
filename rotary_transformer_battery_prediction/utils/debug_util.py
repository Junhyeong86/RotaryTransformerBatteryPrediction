# %%
from typing import Tuple
import torch

def shape_mismatch(tensor: torch.Tensor, expected_shape: Tuple[int, ...]) -> None:
    """
    Check if the shape of a tensor matches the expected shape.
    If not, raise a ValueError with the expected and actual shapes.

    Args:
        tensor (torch.Tensor): The tensor to check.
        expected_shape (tuple): The expected shape of the tensor.

    Raises:
        ValueError: If the shape of the tensor does not match the expected shape.
    """
    if tensor.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch: expected {expected_shape}, got {tensor.shape}"
        )