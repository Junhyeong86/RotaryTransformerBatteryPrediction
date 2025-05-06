# %%
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as f

# %%
class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding

    Attributes:
        embedding_dim: int
            - The dimension of the embedding.
        max_sequence_len: int
            - The maximum sequence length.
        freq_base: float
            - The base frequency for the rotary position embedding.
        dtype: torch.dtype
            - The data type of the rotary position embedding. If None, defaults to torch.float32.
        inverse_freq: torch.Tensor
            - The inverse frequency for the rotary position embedding.
        positions: torch.Tensor
            - The positions for the rotary position embedding.
        angles: torch.Tensor
            - The angles for the rotary position embedding.
        cos: torch.Tensor
            - The cosine values for the rotary position embedding.
        sin: torch.Tensor
            - The sine values for the rotary position embedding.
    """
    def __init__(
            self,
            embedding_dim: int,
            max_sequence_len: int=256,
            freq_base: float = 10000.0,
            dtype: Optional[torch.dtype] = None,
        ):
        """
        Args:
            embedding_dim: int
                - The dimension of the embedding.
            max_sequence_len: int
                - The maximum sequence length.
            freq_base: float
                - The base frequency for the rotary position embedding.
            dtype: torch.dtype
                - The data type of the rotary position embedding. If None, defaults to torch.float32.
        """
        super(RotaryPositionEmbedding, self).__init__()
        assert embedding_dim % 2 == 0, "embedding_dim must be even"
        if dtype is None:
            dtype = torch.float32

        self.embedding_dim = embedding_dim
        self.inverse_freq = 1.0 /(freq_base ** (torch.arange(0, embedding_dim, 2, dtype=dtype) / embedding_dim))
        self.positions = torch.arange(max_sequence_len)

        self.inverse_freq = self.inverse_freq.unsqueeze(0) # (1, embedding_dim // 2)
        self.positions = self.positions.unsqueeze(1) # (max_sequence_len, 1)

        self.angles = self.inverse_freq * self.positions # (max_sequence_len, embedding_dim // 2)
        self.angles = self.angles.repeat_interleave(2, dim=1) # (max_sequence_len, embedding_dim)

        self.cos = torch.cos(self.angles) # (max_sequence_len, embedding_dim)
        self.sin = torch.sin(self.angles) # (max_sequence_len, embedding_dim)

    def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor
                - The input tensor of shape (batch_size, seq_len, embedding_dim).
        Returns:
            torch.Tensor
                - The output tensor of shape (batch_size, seq_len, embedding_dim).
        """
        batch_size, seq_len, _ = x.shape
        # shape_mismatch(x, (batch_size, seq_len, self.embedding_dim))
        assert seq_len <= self.positions.shape[0], f"seq_len {seq_len} exceeds max_sequence_len {self.positions.shape[0]}"

        self.cos = self.cos.to(x.device)
        self.sin = self.sin.to(x.device)

        cos = self.cos[:seq_len, :].unsqueeze(0) # (1, seq_len, embedding_dim)
        sin = self.sin[:seq_len, :].unsqueeze(0) # (1, seq_len, embedding_dim)

        x = x * cos + torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).view(batch_size, seq_len, self.embedding_dim) * sin
        # torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1) -> (batch_size, seq_len, embedding_dim // 2, 2)
        # x2 = x2.view(batch_size, seq_len, self.embedding_dim) -> (batch_size, seq_len, embedding_dim)
        # shape_mismatch(x, (batch_size, seq_len, self.embedding_dim))

        return x

# %%
class RotaryMultiheadAttention(nn.Module):
    """
    Multihead Attention with Rotary Position Embedding

    Attributes:
        embedding_dim: int
            - The dimension of the embedding.
        num_heads: int
            - The number of attention heads.
        dropout: float
            - The dropout rate.
        max_sequence_len: int
            - The maximum sequence length.
        freq_base: float
            - The base frequency for the rotary position embedding.
        rotary_embedding_dtype: torch.dtype
            - The data type of the rotary position embedding. If None, defaults to torch.float32.
        rotary_embedding: RotaryPositionEmbedding
            - The rotary position embedding.
        multihead_attention: nn.MultiheadAttention
            - The multihead attention layer.
    """
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            max_sequence_len: int=256,
            freq_base: float = 10000.0,
            rotary_embedding_dtype: Optional[torch.dtype] = None,
        ):
        """
        Args:
            embedding_dim: int
                - The dimension of the embedding.
            num_heads: int
                - The number of attention heads.
            dropout: float
                - The dropout rate.
            max_sequence_len: int
                - The maximum sequence length.
            freq_base: float
                - The base frequency for the rotary position embedding.
                - LLaMA-2 uses 10000.0 while LLaMA-3 uses 30000.0.
            rotary_embedding_dtype: torch.dtype
                - The data type of the rotary position embedding. If None, defaults to torch.float32.
        """
        super(RotaryMultiheadAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_sequence_len = max_sequence_len
        self.freq_base = freq_base
        self.rotary_embedding_dtype = (
            rotary_embedding_dtype
            if rotary_embedding_dtype is not None
            else torch.float32
        )

        self.rotary_embedding = RotaryPositionEmbedding(
            embedding_dim=self.embedding_dim,
            max_sequence_len=self.max_sequence_len,
            freq_base=self.freq_base,
            dtype=self.rotary_embedding_dtype,
        )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        Args:
            query: torch.Tensor
                - The query tensor of shape (batch_size, query_seq_len, embedding_dim).
            key: torch.Tensor
                - The key tensor of shape (batch_size, key_seq_len, embedding_dim).
            value: torch.Tensor
                - The value tensor of shape (batch_size, key_seq_len, embedding_dim).
            key_padding_mask: Optional[torch.Tensor]
                - The key padding mask of shape (batch_size, key_seq_len).
            attention_mask: Optional[torch.Tensor]
                - The attention mask of shape (query_seq_len, key_seq_len).
        Returns:
            torch.Tensor
                - The output tensor of shape (batch_size, query_seq_len, embedding_dim).
            torch.Tensor
                - The attention weights of shape (batch_size, query_seq_len, key_seq_len).
        """
        batch_size, query_seq_len, _ = query.shape
        _, key_seq_len, _ = key.shape
        # shape_mismatch(query, (batch_size, query_seq_len, self.embedding_dim))
        # shape_mismatch(key, (batch_size, key_seq_len, self.embedding_dim))
        # shape_mismatch(value, (batch_size, key_seq_len, self.embedding_dim))
        # shape_mismatch(key_padding_mask, (batch_size, key_seq_len))
        # shape_mismatch(attention_mask, (query_seq_len, key_seq_len))

        query = self.rotary_embedding(query) # (batch_size, query_seq_len, embedding_dim)
        key = self.rotary_embedding(key) # (batch_size, key_seq_len, embedding_dim)

        if key_padding_mask is not None:
            key_padding_mask = self.boolmask_to_floatmask(
                key_padding_mask,
                dtype=query.dtype,
            )
        if attention_mask is not None:
            attention_mask = self.boolmask_to_floatmask(
                attention_mask,
                dtype=query.dtype,
            )

        ret, ret_attn = self.multihead_attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=attention_mask,
        )
        # ret: (batch_size, query_seq_len, embedding_dim)
        # ret_attn: (batch_size, query_seq_len, key_seq_len)

        return ret, ret_attn

    def boolmask_to_floatmask(
            self,
            mask: torch.Tensor,
            dtype: Optional[torch.dtype] = None,
        ) -> torch.Tensor:
        """
        Convert a boolean mask to a float mask for attention.
        Args:
            mask: torch.Tensor
                - The boolean mask of shape (batch_size, seq_len).
            dtype: Optional[torch.dtype]
                - The data type of the float mask. If None, defaults to torch.float32.
        Returns:
            torch.Tensor
                - The float mask of shape (batch_size, seq_len).
        """
        if dtype is None:
            dtype = torch.float32
        mask_ = torch.full(mask.shape, torch.finfo(dtype).min, dtype=dtype, device=mask.device)
        mask_ = mask_.masked_fill(~mask, 0.0)
        return mask_

# %%
class MultiTimeAttention(nn.Module):
    """
    Multi Time Attention

    Attributes:
        value_dim: int
            - The dimension of the value.
        key_dim: int
            - The dimension of the key.
        embedding_dim: int
            - The dimension of the embedding.
        num_heads: int
            - The number of attention heads.
        ref_seq_len: int
            - The reference time sequence length.
        dropout: float
            - The dropout rate.
    """
    def __init__(
            self,
            value_dim: int,
            key_dim: int,
            embedding_dim: int,
            num_heads: int,
            ref_seq_len: int=16,
            dropout: float = 0.0,
        ):
        """
        Args:
            value_dim: int
                - The dimension of the value.
            key_dim: int
                - The dimension of the key.
            embedding_dim: int
                - The dimension of the embedding.
            num_heads: int
                - The number of attention heads.
            ref_seq_len: int
                - The reference time sequence length.
            dropout: float
                - The dropout rate.
        """
        super(MultiTimeAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.value_dim = value_dim
        self.key_dim = key_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ref_seq_len = ref_seq_len
        self.dropout = dropout

        self.ref_query_seq = torch.arange(self.ref_seq_len)
        self.ref_query_embedding = nn.Embedding(self.ref_seq_len, self.embedding_dim)
        self.key_fc = nn.Linear(self.key_dim, self.embedding_dim)
        self.value_fc = nn.Linear(self.value_dim, self.embedding_dim)

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

    def forward(
            self,
            key_seq: torch.Tensor,
            value_seq: torch.Tensor,
        ) -> torch.Tensor:
        """
        Args:
            key_seq: torch.Tensor
                - The key tensor of shape (batch_size, seq_len, key_dim).
            value_seq: torch.Tensor
                - The value tensor of shape (batch_size, seq_len, value_dim).
        Returns:
            torch.Tensor
                - The output tensor of shape (batch_size, ref_seq_len, embedding_dim).
            torch.Tensor
                - The attention weights of shape (batch_size, ref_seq_len, seq_len).
        """
        batch_size, seq_len, _ = value_seq.shape
        # shape_mismatch(key_seq, (batch_size, seq_len, self.key_dim))
        # shape_mismatch(value_seq, (batch_size, seq_len, self.value_dim))

        self.ref_query_seq = self.ref_query_seq.to(key_seq.device)
        ref_query = self.ref_query_embedding(self.ref_query_seq)
        ref_query = (
            ref_query
            .unsqueeze(0)
            .expand(batch_size, self.ref_seq_len, self.embedding_dim)
        ) # (batch_size, ref_seq_len, embedding_dim)

        key_seq = self.key_fc(key_seq) # (batch_size, seq_len, embedding_dim)
        value_seq = self.value_fc(value_seq) # (batch_size, seq_len, embedding_dim)

        ret, ret_attn = self.multihead_attention(
            query=ref_query,
            key=key_seq,
            value=value_seq,
        )
        # ret: (batch_size, ref_seq_len, embedding_dim)
        # ret_attn: (batch_size, ref_seq_len, seq_len)

        return ret, ret_attn