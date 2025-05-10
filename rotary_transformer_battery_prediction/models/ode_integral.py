# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint

from .feedforward_layers import SwiGLUFeedForward

# %%
class ODEFunc(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            dropout: float = 0.0,
        ):
        super(ODEFunc, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.feed_forward = SwiGLUFeedForward(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )
        self.feed_forward_dropout = nn.Dropout(self.dropout)
        self.norm = nn.RMSNorm(input_dim)

    def forward(
            self,
            t: torch.Tensor,
            x: torch.Tensor
        ) -> torch.Tensor:
        _x = x.clone()

        x = self.norm(x)
        x = self.feed_forward(x)
        x = self.feed_forward_dropout(x)
        x = x + _x

        return x

class ODEIntegral(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float = 0.0,
        ):
        super(ODEIntegral, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.ode_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.ode_layers.append(
                ODEFunc(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    dropout=self.dropout,
                )
            )

    def forward(
            self,
            x0: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
        """
        Args:
            x0: torch.Tensor
                - Initial input tensor of shape (batch_size, input_dim)
            t: torch.Tensor
                - Time tensor of shape (num_steps,)
        Returns:
            torch.Tensor
                - Output tensor of shape (batch_size, num_steps, input_dim)
        """
        x = odeint(
            func=self.ode_layers,
            y0=x0,
            t=t,
            options={"dtype": x0.dtype},
            method="dopri5",
        )
        return x
# %%