# %%
from dataclasses import dataclass
from typing import Optional, List

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
class RoBaTOutput(ModelOutput):
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
        multi_time_attn_weights: Optional[List[torch.FloatTensor]]
            - multi time attention weights
            - length of the list is 1
        time_level_rotary_attn_weights: Optional[List[torch.FloatTensor]]
            - time level rotary attention weights
            - length of the list is equal to the number of time-level transformer layers
        cycle_level_rotary_attn_weights: Optional[List[torch.FloatTensor]]
            - cycle level rotary attention weights
            - length of the list is equal to the number of cycle-level transformer layers
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    discharge_capacity: Optional[torch.FloatTensor] = None
    state_of_health: Optional[torch.FloatTensor] = None
    multi_time_attn_weights: Optional[List[torch.FloatTensor]] = None
    time_level_rotary_attn_weights: Optional[List[torch.FloatTensor]] = None
    cycle_level_rotary_attn_weights: Optional[List[torch.FloatTensor]] = None

# %%
class RoBaTConfig(PretrainedConfig):
    """
    Configuration class for RoBaT.

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
        output_attentions: bool
            - whether to output attention weights
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
            rotary_embedding_dtype: Optional[str]=None,
            max_sequence_len: int=1024,
            output_attentions: bool=False,
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
        self.rotary_embedding_dtype = rotary_embedding_dtype if rotary_embedding_dtype is not None else "float32"
        self.max_sequence_len = max_sequence_len
        self.output_attentions = output_attentions

# %%
class RoBaTPreTrainedModel(PreTrainedModel):
    """
    PreTrainedModel for RoBaT.

    Args:
        config: RoBaTConfig
            - configuration for the model
    """
    config_class = RoBaTConfig
    base_model_prefix = "rotary_transformer_battery_prediction"

    def __init__(self, config: RoBaTConfig):
        super().__init__(config)
        self.muti_time_attn = MultiTimeAttention(
            value_dim=2,
            key_dim=1,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            ref_seq_len=config.ref_time_seq_len,
            dropout=config.dropout
        )
        self.time_transformer_layers = nn.ModuleList()
        for _ in range(config.time_transformer_num_layers):
            self.time_transformer_layers.append(Transformer(
                embedding_dim=config.embedding_dim,
                feed_forward_hidden_dim=config.feed_forward_hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                max_sequence_len=config.ref_time_seq_len,
                freq_base=config.freq_base,
                rotary_embedding_dtype=getattr(torch, config.rotary_embedding_dtype)
            ))
        self.time_norm = nn.RMSNorm(config.embedding_dim)
        self.time_fc = nn.Linear(config.embedding_dim, config.time_transformer_output_dim)

        self.discharge_capacity_embedding = nn.Linear(1, config.embedding_dim)
        self.cycle_transformer_layers = nn.ModuleList()
        for _ in range(config.cycle_transformer_num_layers):
            self.cycle_transformer_layers.append(Transformer(
                embedding_dim=config.embedding_dim,
                feed_forward_hidden_dim=config.feed_forward_hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                max_sequence_len=config.max_sequence_len,
                freq_base=config.freq_base,
                rotary_embedding_dtype=getattr(torch, config.rotary_embedding_dtype),
            ))
        self.cycle_norm = nn.RMSNorm(config.embedding_dim)

        self.output_fc = nn.Linear(config.embedding_dim, 1)
        self.post_init()

    def forward(
            self,
            discharge_capacity: torch.Tensor,
            time: torch.Tensor,
            voltage_per_time: torch.Tensor,
            current_per_time: torch.Tensor,
            mask: Optional[torch.Tensor]=None,
            label_discharge_capacity: Optional[torch.Tensor]=None,
            output_attentions: Optional[bool]=None,
        ) -> RoBaTOutput:
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
            output_attentions: Optional[bool]
                - whether to output attention weights
        Returns:
            RoBaTOutput
                - output of the model
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        if output_attentions:
            multi_time_attn_weights = []
            time_level_rotary_attn_weights = []
            cycle_level_rotary_attn_weights = []
        else:
            multi_time_attn_weights = None
            time_level_rotary_attn_weights = None
            cycle_level_rotary_attn_weights = None

        batch_size, cycle_len = discharge_capacity.shape
        _, _, time_len = time.shape
        # shape_mismatch(discharge_capacity, (batch_size, cycle_len))
        # shape_mismatch(time, (batch_size, cycle_len, time_len))
        # shape_mismatch(voltage_per_time, (batch_size, cycle_len, time_len))
        # shape_mismatch(current_per_time, (batch_size, cycle_len, time_len))
        state_of_health = discharge_capacity / discharge_capacity[:, 0].unsqueeze(1)
        # (batch_size, cycle_len)

        if mask is not None:
            mask = mask.to(state_of_health.device)

        time_features_input = torch.stack([
            voltage_per_time,
            current_per_time,
        ], dim=-1) # (batch_size, cycle_len, time_len, 2)
        time_features_input = time_features_input.view(batch_size * cycle_len, time_len, 2)
        # (batch_size * cycle_len, time_len, 2)
        time_input = time.view(batch_size * cycle_len, time_len, 1)
        # (batch_size * cycle_len, time_len, 1)

        x, _multi_time_attn_weights = self.muti_time_attn(
            key_seq=time_input,
            value_seq=time_features_input,
        ) # (batch_size * cycle_len, ref_time_seq_len, embedding_dim)
        if output_attentions:
            _multi_time_attn_weights = _multi_time_attn_weights.view(
                batch_size, cycle_len, self.config.ref_time_seq_len, time_len
            )
            multi_time_attn_weights.append(_multi_time_attn_weights)

        for layer in self.time_transformer_layers:
            x, _time_level_rotary_attn_weights = layer(
                x=x,
                key_padding_mask=None,
                attention_mask=None,
            ) # (batch_size * cycle_len, ref_time_seq_len, embedding_dim)
            if output_attentions:
                _time_level_rotary_attn_weights = _time_level_rotary_attn_weights.view(
                    batch_size, cycle_len, self.config.ref_time_seq_len, self.config.ref_time_seq_len
                )
                time_level_rotary_attn_weights.append(_time_level_rotary_attn_weights)
        x = self.time_norm(x)
        x = self.time_fc(x) # (batch_size * cycle_len, ref_time_seq_len, time_transformer_output_dim)
        x = x.view(
            batch_size, cycle_len, self.config.embedding_dim
        )

        state_of_health = state_of_health.unsqueeze(2) # (batch_size, cycle_len, 1)
        state_of_health = self.discharge_capacity_embedding(state_of_health) # (batch_size, cycle_len, embedding_dim)
        x = x + state_of_health
        cycle_attn_mask = torch.ones(
            cycle_len, cycle_len, dtype=torch.bool, device=state_of_health.device
        )
        cycle_attn_mask = torch.triu(cycle_attn_mask, diagonal=1)
        for layer in self.cycle_transformer_layers:
            x, _cycle_level_rotary_attn_weights = layer(
                x=x,
                key_padding_mask=mask,
                attention_mask=cycle_attn_mask,
            ) # (batch_size, cycle_len, embedding_dim)
            if output_attentions:
                cycle_level_rotary_attn_weights.append(_cycle_level_rotary_attn_weights)
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

        return RoBaTOutput(
            loss=loss,
            logits=logits,
            discharge_capacity=ret_discharge_capacity,
            state_of_health=ret_state_of_health,
            multi_time_attn_weights=multi_time_attn_weights,
            time_level_rotary_attn_weights=time_level_rotary_attn_weights,
            cycle_level_rotary_attn_weights=cycle_level_rotary_attn_weights,
        )

# %%