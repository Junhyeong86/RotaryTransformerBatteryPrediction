# %%
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import ModelOutput
from transformers import (
    PreTrainedModel,
    PretrainedConfig
)

from .attention_layers import MultiTimeAttention
from .transformer_layer import Transformer

# %%
@dataclass
class BatteryPredictionOutput(ModelOutput):
    """
    Battery Prediction Output

    Attributes:
        loss: Optional[torch.FloatTensor]
            - loss of the model
        logits: Optional[torch.FloatTensor]
            - logits of the model
        discharge_capacity: Optional[torch.FloatTensor]
            - discharge capacity of the battery
        state_of_health: Optional[torch.FloatTensor]
            - state of health predicted by the model
        multi_time_attn_weights: Optional[torch.FloatTensor]
            - multi-time attention weights
        time_level_rotary_attn_weights: Optional[torch.FloatTensor]
            - time level rotary attention weights
        cycle_level_rotary_attn_weights: Optional[torch.FloatTensor]
            - cycle level rotary attention weights
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    discharge_capacity: Optional[torch.FloatTensor] = None
    state_of_health: Optional[torch.FloatTensor] = None
    multi_time_attn_weights: Optional[torch.FloatTensor] = None
    time_level_rotary_attn_weights: Optional[torch.FloatTensor] = None
    cycle_level_rotary_attn_weights: Optional[torch.FloatTensor] = None

# %%
class RotaryTransformerBatteryPrediction(nn.Module):
    """
    LLaMA For Battery Prediction

    Attributes:
        embedding_dim: int
            - embedding dimension
        feed_forward_hidden_dim: int
            - feed forward hidden dimension
        num_heads: int
            - number of heads
        time_transformer_num_layers: int
            - number of transformer layers
        ref_time_seq_len: int
            - reference time sequence length
        time_transformer_output_dim: int
            - time transformer output dimension
        cycle_transformer_num_layers: int
            - number of transformer layers
        dropout: float
            - dropout rate
        freq_base: float
            - frequency base
        rotary_embedding_dtype: Optional[torch.dtype]
            - rotary embedding dtype
        max_sequence_len: int
            - max sequence length
        multi_time_attn: MultiTimeAttention
            - multi time attention layer
        time_transformer_layers: nn.ModuleList
            - time-level transformer layers
        time_norm: nn.RMSNorm
            - time-level RMSNorm layer
        time_fc: nn.Linear
            - time-level linear layer
        discharge_capacity_embedding: nn.Linear
            - discharge capacity embedding layer
        cycle_transformer_layers: nn.ModuleList
            - cycle-level transformer layers
        cycle_norm: nn.RMSNorm
            - cycle-level RMSNorm layer
        output_fc: nn.Linear
            - output linear layer
    """
    def __init__(
            self,
            embedding_dim: int=64,
            feed_forward_hidden_dim: int=128,
            num_heads: int=8,
            time_transformer_num_layers: int=4,
            ref_time_seq_len: int=16,
            time_transformer_output_dim: int=4,
            cycle_transformer_num_layers: int=4,
            dropout: float=0.0,
            freq_base: float=10000.0,
            rotary_embedding_dtype: Optional[str] = None,
            max_sequence_len: int=1024,
        ):
        """
        Args:
            embedding_dim: int
                - embedding dimension
            feed_forward_hidden_dim: int
                - feed forward hidden dimension
            num_heads: int
                - number of heads
            time_transformer_num_layers: int
                - number of transformer layers
            ref_time_seq_len: int
                - reference time sequence length
            time_transformer_output_dim: int
                - time transformer output dimension
            cycle_transformer_num_layers: int
                - number of transformer layers
            dropout: float
                - dropout rate
            freq_base: float
                - frequency base
            rotary_embedding_dtype: Optional[str]
                - rotary embedding dtype
            max_sequence_len: int
                - max sequence length
        """
        super(RotaryTransformerBatteryPrediction, self).__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        assert ref_time_seq_len * time_transformer_output_dim == embedding_dim, "ref_time_seq_len * time_transformer_output_dim must be equal to embedding_dim"

        self.embedding_dim = embedding_dim
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.num_heads = num_heads
        self.time_transformer_num_layers = time_transformer_num_layers
        self.ref_time_seq_len = ref_time_seq_len
        self.time_transformer_output_dim = time_transformer_output_dim
        self.cycle_transformer_num_layers = cycle_transformer_num_layers
        self.dropout = dropout
        self.freq_base = freq_base
        self.rotary_embedding_dtype = (
            getattr(torch, rotary_embedding_dtype)
            if rotary_embedding_dtype is not None
            else torch.float32
        )
        self.max_sequence_len = max_sequence_len

        self.multi_time_attn = MultiTimeAttention(
            value_dim=2,
            key_dim=1,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            ref_seq_len=self.ref_time_seq_len,
            dropout=self.dropout
        )
        self.time_transformer_layers = nn.ModuleList()
        for _ in range(self.time_transformer_num_layers):
            self.time_transformer_layers.append(Transformer(
                embedding_dim=self.embedding_dim,
                feed_forward_hidden_dim=self.feed_forward_hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                max_sequence_len=self.ref_time_seq_len,
                freq_base=self.freq_base,
                rotary_embedding_dtype=self.rotary_embedding_dtype,
            ))
        self.time_norm = nn.RMSNorm(self.embedding_dim)
        self.time_fc = nn.Linear(self.embedding_dim, self.time_transformer_output_dim)

        self.discharge_capacity_embedding = nn.Linear(1, self.embedding_dim)
        self.cycle_transformer_layers = nn.ModuleList()
        for _ in range(self.cycle_transformer_num_layers):
            self.cycle_transformer_layers.append(Transformer(
                embedding_dim=self.embedding_dim,
                feed_forward_hidden_dim=self.feed_forward_hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                max_sequence_len=self.max_sequence_len,
                freq_base=self.freq_base,
                rotary_embedding_dtype=self.rotary_embedding_dtype,
            ))
        self.cycle_norm = nn.RMSNorm(self.embedding_dim)

        self.output_fc = nn.Linear(self.embedding_dim, 1)

    def forward(
            self,
            discharge_capacity: torch.Tensor,
            time: torch.Tensor,
            voltage_per_time: torch.Tensor,
            current_per_time: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            label_discharge_capacity: Optional[torch.Tensor] = None,
        ) -> BatteryPredictionOutput:
        """
        Forward pass of the model.
        Args:
            discharge_capacity: torch.Tensor
                - discharge capacity of the battery
                - shape: (batch_size, cycle_len)
            time: torch.Tensor
                - charge time of the battery
                - shape: (batch_size, cycle_len, time_len)
            voltage_per_time: torch.Tensor
                - charge voltage per time of the battery
                - shape: (batch_size, cycle_len, time_len)
            current_per_time: torch.Tensor
                - charge current per time of the battery
                - shape: (batch_size, cycle_len, time_len)
            mask: Optional[torch.Tensor]
                - mask for the input data
            label_discharge_capacity: Optional[torch.Tensor]
                - label discharge capacity for loss calculation
        Returns:
            BatteryPredictionOutput
                - output of the model
        """
        batch_size, cycle_len = discharge_capacity.shape
        _, _, time_len = time.shape
        # shape_mismatch(discharge_capacity, (batch_size, cycle_len))
        # shape_mismatch(time, (batch_size, cycle_len, time_len))
        # shape_mismatch(voltage_per_time, (batch_size, cycle_len, time_len))
        # shape_mismatch(current_per_time, (batch_size, cycle_len, time_len))
        state_of_health = discharge_capacity / discharge_capacity[:, 0].unsqueeze(1) # (batch_size, cycle_len)

        if mask is not None:
            mask = mask.to(state_of_health.device)

        time_features_input = torch.stack([
            voltage_per_time,
            current_per_time,
        ], dim=-1) # (batch_size, cycle_len, time_len, 2)
        time_features_input = time_features_input.view(batch_size * cycle_len, time_len, 2) # (batch_size * cycle_len, time_len, 2)
        time_input = time.view(batch_size * cycle_len, time_len, 1) # (batch_size * cycle_len, time_len, 1)

        x, multi_time_attn_weights = self.multi_time_attn(
            key_seq=time_input,
            value_seq=time_features_input,
        ) # (batch_size * cycle_len, ref_time_seq_len, embedding_dim)
        for layer in self.time_transformer_layers:
            x, time_level_rotary_attn_weights = layer(
                x=x,
                key_padding_mask=None,
                attention_mask=None,
            ) # (batch_size * cycle_len, ref_time_seq_len, embedding_dim)
        x = self.time_norm(x)
        x = self.time_fc(x) # (batch_size * cycle_len, ref_time_seq_len, time_transformer_output_dim)
        x = x.view(batch_size, cycle_len, self.embedding_dim)
        multi_time_attn_weights = multi_time_attn_weights.view(batch_size, cycle_len, self.ref_time_seq_len, time_len)
        time_level_rotary_attn_weights = time_level_rotary_attn_weights.view(batch_size, cycle_len, self.ref_time_seq_len, self.ref_time_seq_len)

        state_of_health = state_of_health.unsqueeze(2) # (batch_size, cycle_len, 1)
        state_of_health = self.discharge_capacity_embedding(state_of_health) # (batch_size, cycle_len, embedding_dim)
        x = x + state_of_health
        cycle_attn_mask = torch.ones(cycle_len, cycle_len, dtype=torch.bool, device=state_of_health.device)
        cycle_attn_mask = torch.triu(cycle_attn_mask, diagonal=1)
        for layer in self.cycle_transformer_layers:
            x, cycle_level_rotary_attn_weights = layer(
                x=x,
                key_padding_mask=mask,
                attention_mask=cycle_attn_mask,
            ) # (batch_size, cycle_len, embedding_dim)
        x = self.cycle_norm(x)

        logits = self.output_fc(x) # (batch_size, cycle_len, 1)
        logits = logits.squeeze(2) # (batch_size, cycle_len)

        ret_state_of_health = logits / logits[:, 0].unsqueeze(1) # (batch_size, cycle_len)
        ret_discharge_capacity = ret_state_of_health * discharge_capacity[:, 0].unsqueeze(1) # (batch_size, cycle_len)

        loss = None
        if label_discharge_capacity is not None:
            label_state_of_health = label_discharge_capacity / label_discharge_capacity[:, 0].unsqueeze(1)
            loss = F.mse_loss(
                ret_state_of_health,
                label_state_of_health
            ) * 100 # to percentage loss of state of health

        ret = BatteryPredictionOutput(
            loss=loss,
            logits=logits,
            discharge_capacity=ret_discharge_capacity,
            state_of_health=ret_state_of_health,
            multi_time_attn_weights=multi_time_attn_weights,
            time_level_rotary_attn_weights=time_level_rotary_attn_weights,
            cycle_level_rotary_attn_weights=cycle_level_rotary_attn_weights,
        )
        return ret

# %%
class RotaryTransformerBatteryPredictionConfig(PretrainedConfig):
    """
    Configuration class for RotaryTransformerBatteryPrediction.

    Args:
        embedding_dim: int
            - embedding dimension
        feed_forward_hidden_dim: int
            - feed forward hidden dimension
        num_heads: int
            - number of heads
        time_transformer_num_layers: int
            - number of transformer layers
        ref_time_seq_len: int
            - reference time sequence length
        time_transformer_output_dim: int
            - time transformer output dimension
        cycle_transformer_num_layers: int
            - number of transformer layers
        dropout: float
            - dropout rate
        freq_base: float
            - frequency base
        rotary_embedding_dtype: Optional[torch.dtype]
            - rotary embedding dtype
        max_sequence_len: int
            - max sequence length
    """
    model_type = "rotary_transformer_battery_prediction"

    def __init__(
            self,
            embedding_dim: int=64,
            feed_forward_hidden_dim: int=128,
            num_heads: int=8,
            time_transformer_num_layers: int=4,
            ref_time_seq_len: int=16,
            time_transformer_output_dim: int=4,
            cycle_transformer_num_layers: int=4,
            dropout: float=0.0,
            freq_base: float=10000.0,
            rotary_embedding_dtype: Optional[torch.dtype] = None,
            max_sequence_len: int=1024,
        ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.num_heads = num_heads
        self.time_transformer_num_layers = time_transformer_num_layers
        self.ref_time_seq_len = ref_time_seq_len
        self.time_transformer_output_dim = time_transformer_output_dim
        self.cycle_transformer_num_layers = cycle_transformer_num_layers
        self.dropout = dropout
        self.freq_base = freq_base
        self.rotary_embedding_dtype = (
            rotary_embedding_dtype if rotary_embedding_dtype is not None else torch.float32)
        self.max_sequence_len = max_sequence_len

# %%
class RotaryTransformerBatteryPredictionPreTrainedModel(PreTrainedModel):
    """
    PreTrainedModel for RotaryTransformerBatteryPrediction.

    Args:
        config: RotaryTransformerBatteryPredictionConfig
            - configuration for the model
    """
    config_class = RotaryTransformerBatteryPredictionConfig
    base_model_prefix = "rotary_transformer_battery_prediction"

    def __init__(self, config: RotaryTransformerBatteryPredictionConfig):
        super().__init__(config)
        self.model = RotaryTransformerBatteryPrediction(
            embedding_dim=config.embedding_dim,
            feed_forward_hidden_dim=config.feed_forward_hidden_dim,
            num_heads=config.num_heads,
            time_transformer_num_layers=config.time_transformer_num_layers,
            ref_time_seq_len=config.ref_time_seq_len,
            time_transformer_output_dim=config.time_transformer_output_dim,
            cycle_transformer_num_layers=config.cycle_transformer_num_layers,
            dropout=config.dropout,
            freq_base=config.freq_base,
            rotary_embedding_dtype=config.rotary_embedding_dtype,
            max_sequence_len=config.max_sequence_len,
        )
        self.post_init()

    def forward(
            self,
            discharge_capacity: torch.Tensor,
            time: torch.Tensor,
            voltage_per_time: torch.Tensor,
            current_per_time: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            label_discharge_capacity: Optional[torch.Tensor] = None,
        ) -> BatteryPredictionOutput:
        """
        Forward pass of the model.
        Args:
            discharge_capacity: torch.Tensor
                - discharge capacity of the battery
                - shape: (batch_size, cycle_len)
            time: torch.Tensor
                - charge time of the battery
                - shape: (batch_size, cycle_len, time_len)
            voltage_per_time: torch.Tensor
                - charge voltage per time of the battery
                - shape: (batch_size, cycle_len, time_len)
            current_per_time: torch.Tensor
                - charge current per time of the battery
                - shape: (batch_size, cycle_len, time_len)
            mask: Optional[torch.Tensor]
                - mask for the input data
            label_discharge_capacity: Optional[torch.Tensor]
                - label discharge capacity for loss calculation
        Returns:
            BatteryPredictionOutput
                - output of the model
        """
        return self.model(
            discharge_capacity=discharge_capacity,
            time=time,
            voltage_per_time=voltage_per_time,
            current_per_time=current_per_time,
            mask=mask,
            label_discharge_capacity=label_discharge_capacity,
        )