# %%
import torch
import torch.nn as nn
import torch.nn.functional as f

# %%
class SwiGLUFeedForward(nn.Module):
    """
    LLaMA Feed Forward Network is SwiGLU activation function

    Attributes:
        input_dim: int
            - Input dimension
        hidden_dim: int
            - Hidden dimension
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
        ):
        """
        Args:
            input_dim: int
                - Input dimension
            hidden_dim: int
                - Hidden dimension
        """
        super(SwiGLUFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(
            self,
            x: torch.Tensor
        ) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor
                - Input tensor of shape (..., input_dim)
        Returns:
            torch.Tensor
                - Output tensor of shape (..., input_dim)
        """
        x_fc1 = self.fc1(x) # (..., hidden_dim)
        x_fc2 = self.fc2(x) # (..., hidden_dim)
        x = self.silu(x_fc1) * x_fc2 # (..., hidden_dim)
        x = self.fc3(x) # (..., input_dim)
        return x