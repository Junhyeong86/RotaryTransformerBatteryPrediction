# %%
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from logging import getLogger

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
            embedding_or_head_dim: int,
            max_sequence_len: int=256,
            freq_base: float = 10000.0,
            dtype: Optional[torch.dtype] = None,
        ):
        """
        Args:
            embedding_or_head_dim: int
                - The dimension of the embedding or head.
            max_sequence_len: int
                - The maximum sequence length.
            freq_base: float
                - The base frequency for the rotary position embedding.
            dtype: torch.dtype
                - The data type of the rotary position embedding. If None, defaults to torch.float32.
        """
        super(RotaryPositionEmbedding, self).__init__()
        assert embedding_or_head_dim % 2 == 0, "embedding_or_head_dim must be even"
        if dtype is None:
            dtype = torch.float32

        self.embedding_or_head_dim = embedding_or_head_dim
        self.inverse_freq = 1.0 / (freq_base ** (torch.arange(0, embedding_or_head_dim, 2, dtype=dtype) / embedding_or_head_dim))
        self.positions = torch.arange(max_sequence_len)

        self.inverse_freq = self.inverse_freq.unsqueeze(0) # (1, embedding_or_head_dim // 2)
        self.positions = self.positions.unsqueeze(1) # (max_sequence_len, 1)

        self.angles = self.inverse_freq * self.positions # (max_sequence_len, embedding_dim // 2)
        self.angles = self.angles.repeat_interleave(2, dim=1) # (max_sequence_len, embedding_dim)

        self.cos = torch.cos(self.angles) # (max_sequence_len, embedding_dim)
        self.sin = torch.sin(self.angles) # (max_sequence_len, embedding_dim)

        self.logger = getLogger(__name__ + "." + __class__.__name__)

    def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor
                - The input tensor of shape (batch_size, num_heads, seq_len, head_dim)
                - or (batch_size, seq_len, embedding_dim).
        Returns:
            torch.Tensor
                - The output tensor of shape (batch_size, seq_len, embedding_dim)
                - or (batch_size, num_heads, seq_len, head_dim).
        """
        self.logger.debug(
            f"RotaryPositionEmbedding forward called with:\n"
            f"  x shape: {x.shape}\n"
        )
        self.cos = self.cos.to(x.device)
        self.sin = self.sin.to(x.device)

        if x.dim() == 4:
            batch_size, num_heads, seq_len, _ = x.shape
            cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, embedding_or_head_dim)
            sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, embedding_or_head_dim)
            x = (
                cos * x
                + sin * torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).view(
                    batch_size, num_heads, seq_len, self.embedding_or_head_dim
                )
            )
            # torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1) -> (batch_size, num_heads, seq_len, embedding_or_head_dim // 2, 2)
        elif x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            cos = self.cos[:seq_len, :].unsqueeze(0) # (1, seq_len, embedding_or_head_dim)
            sin = self.sin[:seq_len, :].unsqueeze(0) # (1, seq_len, embedding_or_head_dim)
            x = (
                cos * x
                + sin * torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).view(
                    batch_size, seq_len, self.embedding_or_head_dim
                )
            )
            # torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1) -> (batch_size, seq_len, embedding_or_head_dim // 2, 2)
        else:
            raise ValueError("Input tensor x must be of shape (batch_size, num_heads, seq_len, head_dim) or (batch_size, seq_len, embedding_dim).")

        self.logger.debug(
            f"RotaryPositionEmbedding forward output:\n"
            f"  x shape: {x.shape}\n"
        )
        assert seq_len <= self.positions.shape[0], f"seq_len {seq_len} exceeds max_sequence_len {self.positions.shape[0]}"
        return x

# %%
def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float=0.0
    ):
    logger = getLogger(__name__ + ".eager_attention_forward")
    logger.debug(
        f"eager_attention_forward called with:\n"
        f"  query shape: {query.shape}\n"
        f"  key shape: {key.shape}\n"
        f"  value shape: {value.shape}\n"
        f"  attention_mask shape: {attention_mask.shape if attention_mask is not None else None}\n"
        f"  padding_mask shape: {padding_mask.shape if padding_mask is not None else None}\n"
        f"  scaling: {scaling}\n"
        f"  dropout: {dropout}\n"
    )
    batch_size, num_heads, query_seq_len, head_dim = query.shape
    _, _, key_seq_len, _ = key.shape
    logger.debug(
        f"Input query, key, value tensors...\n"
        f"  expected input query shape: (batch_size, num_heads, query_seq_len, head_dim)\n"
        f"  actual input query shape: {query.shape}\n"
        f"  expected input key shape: (batch_size, num_heads, key_seq_len, head_dim)\n"
        f"  actual input key shape: {key.shape}\n"
        f"  expected input value shape: (batch_size, num_heads, key_seq_len, head_dim)\n"
        f"  actual input value shape: {value.shape}\n"
    )
    mask = torch.zeros((batch_size, num_heads, query_seq_len, key_seq_len), device=query.device)
    if attention_mask is not None:
        logger.debug(
            f"Applying attention mask...\n"
            f"  expected attention_mask shape: (query_seq_len, key_seq_len)\n"
            f"  actual attention_mask shape: {attention_mask.shape}\n"
        )
        mask = mask + attention_mask.unsqueeze(0).unsqueeze(0) # (batch_size, num_heads, query_seq_len, key_seq_len)
    if padding_mask is not None:
        logger.debug(
            f"Applying padding mask...\n"
            f"  expected padding_mask shape: (batch_size, key_seq_len)\n"
            f"  actual padding_mask shape: {padding_mask.shape}\n"
        )
        mask = mask + padding_mask.unsqueeze(1).unsqueeze(1) # (batch_size, num_heads, query_seq_len, key_seq_len)

    logger.debug(
        f"Computing attention weights...\n"
        f"  expected input query shape: (batch_size, num_heads, query_seq_len, head_dim)\n"
        f"  actual input query shape: {query.shape}\n"
        f"  expected input key shape: (batch_size, num_heads, key_seq_len, head_dim)\n"
        f"  actual input key shape: {key.shape}\n"
    )
    attn_weights = torch.matmul(
        query, # (batch_size, num_heads, query_seq_len, head_dim)
        key.permute(0, 1, 3, 2) # (batch_size, num_heads, head_dim, key_seq_len)
    ) * scaling # (batch_size, num_heads, query_seq_len, key_seq_len)
    logger.debug(
        f"Computed attention weights...\n"
        f"  expected output attn_weights shape: (batch_size, num_heads, query_seq_len, key_seq_len)\n"
        f"  actual output attn_weights shape: {attn_weights.shape}\n"
    )

    attn_weights = attn_weights + mask # (batch_size, num_heads, query_seq_len, key_seq_len)
    attn_weights = F.softmax(attn_weights, dim=-1) # (batch_size, num_heads, query_seq_len, key_seq_len)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(
        attn_weights, # (batch_size, num_heads, query_seq_len, key_seq_len)
        value # (batch_size, num_heads, key_seq_len, head_dim)
    ) # (batch_size, num_heads, query_seq_len, head_dim)
    logger.debug(
        f"Computed attention output...\n"
        f"  expected output attn_output shape: (batch_size, num_heads, query_seq_len, head_dim)\n"
        f"  actual output attn_output shape: {attn_output.shape}\n"
    )
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous() # (batch_size, query_seq_len, num_heads, head_dim)

    ret_attn_weights = attn_weights.sum(dim=1) / module.num_heads # (batch_size, query_seq_len, key_seq_len)
    return attn_output, ret_attn_weights

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
        self.head_dim = embedding_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout
        self.max_sequence_len = max_sequence_len
        self.freq_base = freq_base
        self.rotary_embedding_dtype = (
            rotary_embedding_dtype
            if rotary_embedding_dtype is not None
            else torch.float32
        )

        self.rotary_embedding = RotaryPositionEmbedding(
            embedding_or_head_dim=self.head_dim,
            max_sequence_len=self.max_sequence_len,
            freq_base=self.freq_base,
            dtype=self.rotary_embedding_dtype,
        )

        self.q_proj = nn.Linear(
            self.embedding_dim,
            self.embedding_dim
        )
        self.k_proj = nn.Linear(
            self.embedding_dim,
            self.embedding_dim
        )
        self.v_proj = nn.Linear(
            self.embedding_dim,
            self.embedding_dim
        )
        self.o_proj = nn.Linear(
            self.embedding_dim,
            self.embedding_dim
        )

        self.logger = getLogger(__name__ + "." + __class__.__name__)

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
        self.logger.debug(
            f"RotaryMultiheadAttention forward called with:\n"
            f"  query shape: {query.shape}\n"
            f"  key shape: {key.shape}\n"
            f"  value shape: {value.shape}\n"
            f"  key_padding_mask shape: {key_padding_mask.shape if key_padding_mask is not None else None}\n"
            f"  attention_mask shape: {attention_mask.shape if attention_mask is not None else None}\n"
        )
        # shape_mismatch(query, (batch_size, query_seq_len, self.embedding_dim))
        # shape_mismatch(key, (batch_size, key_seq_len, self.embedding_dim))
        # shape_mismatch(value, (batch_size, key_seq_len, self.embedding_dim))
        # shape_mismatch(key_padding_mask, (batch_size, key_seq_len))
        # shape_mismatch(attention_mask, (query_seq_len, key_seq_len))

        query: torch.Tensor = self.q_proj(query)
        key: torch.Tensor = self.k_proj(key)
        value: torch.Tensor = self.v_proj(value)

        query = query.view(
            batch_size, query_seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3) # (batch_size, self.num_heads, query_seq_len, self.head_dim)
        key = key.view(
            batch_size, key_seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3) # (batch_size, self.num_heads, key_seq_len, self.head_dim)
        value = value.view(
            batch_size, key_seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3) # (batch_size, self.num_heads, key_seq_len, self.head_dim)

        query = self.rotary_embedding(query) # (batch_size, self.num_heads, query_seq_len, head_dim)
        key = self.rotary_embedding(key) # (batch_size, self.num_heads, key_seq_len, head_dim)

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

        attn_output, attn_weights = eager_attention_forward(
            module=self,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            padding_mask=key_padding_mask,
            scaling=self.scaling,
            dropout=self.dropout
        )

        attn_output = attn_output.view(
            batch_size,
            query_seq_len,
            self.embedding_dim
        ).contiguous() # (batch_size, query_seq_len, embedding_dim)
        attn_output = self.o_proj(attn_output) # (batch_size, query_seq_len, embedding_dim)

        # attn_output: (batch_size, query_seq_len, embedding_dim)
        # attn_weights: (batch_size, query_seq_len, key_seq_len)
        return attn_output, attn_weights

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
        self.head_dim = embedding_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.ref_seq_len = ref_seq_len
        self.dropout = dropout

        self.ref_query_seq = torch.arange(self.ref_seq_len)
        self.ref_query_embedding = nn.Embedding(self.ref_seq_len, self.embedding_dim)
        self.key_fc = nn.Linear(self.key_dim, self.embedding_dim)
        self.value_fc = nn.Linear(self.value_dim, self.embedding_dim)

        self.q_proj = nn.Linear(
            self.embedding_dim,
            self.embedding_dim
        )
        self.k_proj = nn.Linear(
            self.embedding_dim,
            self.embedding_dim
        )
        self.v_proj = nn.Linear(
            self.embedding_dim,
            self.embedding_dim
        )
        self.o_proj = nn.Linear(
            self.embedding_dim,
            self.embedding_dim
        )

        self.logger = getLogger(__name__ + "." + __class__.__name__)

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
        self.logger.debug(
            f"MultiTimeAttention forward called with:\n"
            f"  key_seq shape: {key_seq.shape}\n"
            f"  value_seq shape: {value_seq.shape}\n"
        )
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

        query: torch.Tensor = self.q_proj(ref_query)
        key: torch.Tensor = self.k_proj(key_seq)
        value: torch.Tensor = self.v_proj(value_seq)

        query = query.view(
            batch_size, self.ref_seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3) # (batch_size, self.num_heads, ref_seq_len, head_dim)
        key = key.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3) # (batch_size, self.num_heads, seq_len, head_dim)
        value = value.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3) # (batch_size, self.num_heads, seq_len, head_dim)

        attn_output, attn_weights = eager_attention_forward(
            module=self,
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            padding_mask=None,
            scaling=self.scaling,
            dropout=self.dropout
        )

        attn_output = attn_output.view(
            batch_size,
            self.ref_seq_len,
            self.embedding_dim
        ).contiguous() # (batch_size, ref_seq_len, embedding_dim)
        attn_output = self.o_proj(attn_output) # (batch_size, ref_seq_len, embedding_dim)

        # attn_output: (batch_size, ref_seq_len, embedding_dim)
        # attn_weights: (batch_size, ref_seq_len, seq_len)
        return attn_output, attn_weights

# %%