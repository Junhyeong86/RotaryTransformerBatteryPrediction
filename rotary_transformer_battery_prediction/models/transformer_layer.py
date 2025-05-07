# %%
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as f

from .attention_layers import RotaryMultiheadAttention
from .feedforward_layers import SwiGLUFeedForward

# %%
class Transformer(nn.Module):
    """
    Transformer Layer with Rotary Multihead Attention and SwiGLU Feed Forward Network

    Attributes:
        embedding_dim: int
            - Embedding dimension
        feed_forward_hidden_dim: int
            - Feed forward hidden dimension
        num_heads: int
            - Number of attention heads
        dropout: float
            - Dropout rate
        max_sequence_len: int
            - Maximum sequence length
        freq_base: float
            - Frequency base for rotary embeddings
        rotary_embedding_dtype: Optional[torch.dtype]
            - Data type for rotary embeddings
    """
    def __init__(
            self,
            embedding_dim: int,
            feed_forward_hidden_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            max_sequence_len: int=256,
            freq_base: float = 10000.0,
            rotary_embedding_dtype: Optional[torch.dtype] = None,
        ):
        """
        Args:
            embedding_dim: int
                - Embedding dimension
            feed_forward_hidden_dim: int
                - Feed forward hidden dimension
            num_heads: int
                - Number of attention heads
            dropout: float
                - Dropout rate
            max_sequence_len: int
                - Maximum sequence length
            freq_base: float
                - Frequency base for rotary embeddings
            rotary_embedding_dtype: Optional[torch.dtype]
                - Data type for rotary embeddings
        """
        super(Transformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_sequence_len = max_sequence_len
        self.freq_base = freq_base
        self.rotary_embedding_dtype = (
            rotary_embedding_dtype
            if rotary_embedding_dtype is not None
            else torch.float32
        )

        self.attention = RotaryMultiheadAttention(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            max_sequence_len=self.max_sequence_len,
            freq_base=self.freq_base,
            rotary_embedding_dtype=self.rotary_embedding_dtype,
        )
        self.feed_forward = SwiGLUFeedForward(
            input_dim=self.embedding_dim,
            hidden_dim=self.feed_forward_hidden_dim,
        )
        self.feed_forward_dropout = nn.Dropout(self.dropout)
        self.norm1 = nn.RMSNorm(self.embedding_dim)
        self.norm2 = nn.RMSNorm(self.embedding_dim)

    def forward(
            self,
            x: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor
                - Input tensor of shape (batch_size, seq_len, embedding_dim)
            key_padding_mask: Optional[torch.Tensor]
                - Key padding mask of shape (batch_size, seq_len)
            attention_mask: Optional[torch.Tensor]
                - Attention mask of shape (seq_len, seq_len)
        Returns:
            torch.Tensor
                - Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, _ = x.shape
        # shape_mismatch(x, (batch_size, seq_len, self.embedding_dim))
        # shape_mismatch(key_padding_mask, (batch_size, seq_len))
        # shape_mismatch(attention_mask, (seq_len, seq_len))

        _x = x.clone()

        x = self.norm1(x)
        x, attn = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )
        x = x + _x

        _x = x.clone()
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.feed_forward_dropout(x)
        x = x + _x

        return x, attn

# %%