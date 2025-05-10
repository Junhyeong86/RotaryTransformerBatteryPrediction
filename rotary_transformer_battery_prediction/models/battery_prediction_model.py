# %%
from dataclasses import dataclass
from typing import Optional, List, Literal

import numpy

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
from .ode_integral import ODEIntegral

# %%
@dataclass
class RoBaTOutput(ModelOutput):
    """
    Battery Prediction Output for State of Health Prediction

    Attributes:
        loss: Optional[torch.FloatTensor]
            - loss of the model
        discharge_capacity: Optional[torch.FloatTensor]
            - discharge capacity of the battery
        state_of_health: Optional[torch.FloatTensor]
            - state of health predicted by the model
        last_hidden_state: Optional[torch.FloatTensor]
            - last hidden state of the model
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
    discharge_capacity: Optional[torch.FloatTensor] = None
    state_of_health: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
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
        initializer_range: float
            - initializer range
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
            ode_num_layers: int=4,
            ode_prediction_len: int=128,
            ode_step_dtype: str="float32",
            dropout: float=0.0,
            freq_base: float=10000.0,
            loss_column: Literal["discharge_capacity", "state_of_health"] = "state_of_health",
            rotary_embedding_dtype: str="float32",
            max_sequence_len: int=1024,
            initializer_range: float=0.02,
            output_attentions: bool=False,
            padding_value: int=-100,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.num_heads = num_heads
        self.time_transformer_num_layers = time_transformer_num_layers
        self.ref_time_seq_len = ref_time_seq_len
        self.time_transformer_output_dim = time_transformer_output_dim
        self.cycle_transformer_num_layers = cycle_transformer_num_layers
        self.ode_num_layers = ode_num_layers
        self.ode_prediction_len = ode_prediction_len
        self.ode_step_dtype = ode_step_dtype
        self.dropout = dropout
        self.freq_base = freq_base
        self.loss_column = loss_column
        self.rotary_embedding_dtype = rotary_embedding_dtype
        self.max_sequence_len = max_sequence_len
        self.initializer_range = initializer_range
        self.output_attentions = output_attentions
        self.padding_value = padding_value

# %%
class RoBaTPreTrainedModel(PreTrainedModel):
    config_class = RoBaTConfig
    base_model_prefix = "rotary_transformer_battery_prediction"
    supports_gradient_checkpointing = True

    def __init__(self, config: RoBaTConfig):
        super().__init__(config)
        self.config = config

    def _init_weights(self, module):
        """
        Initialize the weights of the model.

        Reference:
            https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

        Args:
            module: nn.Module
                - module to initialize
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)


    def _calc_loss(
            self,
            ret_discharge_capacity: torch.Tensor,
            label_discharge_capacity: torch.Tensor,
            ret_state_of_health: torch.Tensor,
            label_state_of_health: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
        """
        Calculate loss for the model.
        Args:
            ret_discharge_capacity: torch.Tensor
                - predicted discharge capacity
                - shape: (batch_size, cycle_len)
            label_discharge_capacity: torch.Tensor
                - label discharge capacity
                - shape: (batch_size, cycle_len)
            ret_state_of_health: torch.Tensor
                - predicted state of health
                - shape: (batch_size, cycle_len)
            label_state_of_health: torch.Tensor
                - label state of health
                - shape: (batch_size, cycle_len)
            mask: torch.Tensor
                - mask for the input data
                - shape: (batch_size, cycle_len)
        Returns:
            torch.Tensor
                - loss of the model
        """
        ret_discharge_capacity = ret_discharge_capacity.masked_fill(mask, self.config.padding_value)
        ret_state_of_health = ret_state_of_health.masked_fill(mask, self.config.padding_value)

        loss = None
        if label_discharge_capacity is not None and self.config.loss_column == "discharge_capacity":
            loss = F.mse_loss(
                ret_discharge_capacity,
                label_discharge_capacity,
                reduction="sum"
            ) / (~mask).sum() * 1e6
        elif label_state_of_health is not None and self.config.loss_column == "state_of_health":
            loss = F.mse_loss(
                ret_state_of_health,
                label_state_of_health,
                reduction="sum"
            ) / (~mask).sum() * 1e4
        return loss

# %%
class RoBaTModel(RoBaTPreTrainedModel):
    """
    RoBaT Model

    Args:
        config: RoBaTConfig
            - configuration for the model
    """
    def __init__(self, config: RoBaTConfig):
        super().__init__(config)

        self.multi_time_attn = MultiTimeAttention(
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

        self.post_init()

    def forward(
            self,
            discharge_capacity: torch.Tensor,
            first_discharge_capacity: torch.Tensor,
            state_of_health: torch.Tensor,
            time: torch.Tensor,
            voltage_per_time: torch.Tensor,
            current_per_time: torch.Tensor,
            mask: Optional[torch.Tensor]=None,
            label_discharge_capacity: Optional[torch.Tensor]=None,
            label_state_of_health: Optional[torch.Tensor]=None,
            output_attentions: Optional[bool]=None,
        ) -> RoBaTOutput:
        """
        Forward pass of the model.
        Args:
            discharge_capacity: torch.Tensor
                - discharge capacity of the battery
                - shape: (batch_size, cycle_len)
            first_discharge_capacity: torch.Tensor
                - first discharge capacity of the battery
                - shape: (batch_size, cycle_len)
            state_of_health: torch.Tensor
                - state of health of the battery
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
                - shape: (batch_size, cycle_len)
            label_discharge_capacity: Optional[torch.Tensor]
                - label discharge capacity for loss calculation
                - shape: (batch_size, cycle_len)
            label_state_of_health: Optional[torch.Tensor]
                - label state of health for loss calculation
                - shape: (batch_size, cycle_len)
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
        # shape_mismatch(first_discharge_capacity, (batch_size, cycle_len))
        # shape_mismatch(state_of_health, (batch_size, cycle_len))
        # shape_mismatch(time, (batch_size, cycle_len, time_len))
        # shape_mismatch(voltage_per_time, (batch_size, cycle_len, time_len))
        # shape_mismatch(current_per_time, (batch_size, cycle_len, time_len))
        # if label_discharge_capacity is not None:
            # shape_mismatch(label_discharge_capacity, (batch_size, cycle_len))
        # if label_state_of_health is not None:
            # shape_mismatch(label_state_of_health, (batch_size, cycle_len))
        # (batch_size, cycle_len)

        time_features_input = torch.stack([
            voltage_per_time,
            current_per_time,
        ], dim=-1) # (batch_size, cycle_len, time_len, 2)
        time_features_input = time_features_input.view(batch_size * cycle_len, time_len, 2)
        # (batch_size * cycle_len, time_len, 2)
        time_input = time.view(batch_size * cycle_len, time_len, 1)
        # (batch_size * cycle_len, time_len, 1)

        x, _multi_time_attn_weights = self.multi_time_attn(
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

        return RoBaTOutput(
            loss=None,
            discharge_capacity=None,
            state_of_health=None,
            last_hidden_state=x,
            multi_time_attn_weights=multi_time_attn_weights,
            time_level_rotary_attn_weights=time_level_rotary_attn_weights,
            cycle_level_rotary_attn_weights=cycle_level_rotary_attn_weights,
        )

# %%
class RoBaTForNextCyclePrediction(RoBaTPreTrainedModel):
    """
    RoBaT for Next Cycle Prediction

    Args:
        config: RoBaTConfig
            - configuration for the model
    """
    def __init__(self, config: RoBaTConfig):
        super().__init__(config)
        self.model = RoBaTModel(config)

        self.next_cycle_regressor = nn.Linear(config.embedding_dim, 1)

        self.post_init()

    def forward(
            self,
            discharge_capacity: torch.Tensor,
            first_discharge_capacity: torch.Tensor,
            state_of_health: torch.Tensor,
            time: torch.Tensor,
            voltage_per_time: torch.Tensor,
            current_per_time: torch.Tensor,
            mask: Optional[torch.Tensor]=None,
            label_discharge_capacity: Optional[torch.Tensor]=None,
            label_state_of_health: Optional[torch.Tensor]=None,
            output_attentions: Optional[bool]=None,
        ) -> RoBaTOutput:

        output = self.model(
            discharge_capacity=discharge_capacity,
            first_discharge_capacity=first_discharge_capacity,
            state_of_health=state_of_health,
            time=time,
            voltage_per_time=voltage_per_time,
            current_per_time=current_per_time,
            mask=mask,
            label_discharge_capacity=label_discharge_capacity,
            label_state_of_health=label_state_of_health,
            output_attentions=output_attentions
        )
        x = output.last_hidden_state # (batch_size, cycle_len, embedding_dim)

        ret_state_of_health = self.next_cycle_regressor(x) # (batch_size, cycle_len, 1)
        ret_state_of_health = ret_state_of_health.squeeze(2) # (batch_size, cycle_len)

        ret_discharge_capacity = ret_state_of_health * first_discharge_capacity.unsqueeze(1) # (batch_size, cycle_len)

        ret_discharge_capacity = ret_discharge_capacity.masked_fill(mask, self.config.padding_value)
        ret_state_of_health = ret_state_of_health.masked_fill(mask, self.config.padding_value)

        loss = self._calc_loss(
            ret_discharge_capacity=ret_discharge_capacity,
            label_discharge_capacity=label_discharge_capacity,
            ret_state_of_health=ret_state_of_health,
            label_state_of_health=label_state_of_health,
            mask=mask
        )

        return RoBaTOutput(
            loss=loss,
            discharge_capacity=ret_discharge_capacity,
            state_of_health=ret_state_of_health,
            multi_time_attn_weights=output.multi_time_attn_weights,
            time_level_rotary_attn_weights=output.time_level_rotary_attn_weights,
            cycle_level_rotary_attn_weights=output.cycle_level_rotary_attn_weights,
        )

# %%
class RoBaTForTrajectoryPrediction(RoBaTPreTrainedModel):
    """
    RoBaT for Discharge Capacity Prediction

    Args:
        config: RoBaTConfig
            - configuration for the model
    """
    def __init__(self, config: RoBaTConfig):
        super().__init__(config)
        self.model = RoBaTModel(config)
        self.ode_integral = ODEIntegral(
            input_dim=config.embedding_dim,
            hidden_dim=config.feed_forward_hidden_dim,
            num_layers=config.ode_num_layers,
            dropout=config.dropout,
        )
        self.ode_step = torch.arange(
            0,
            config.ode_prediction_len,
            dtype=getattr(torch, config.ode_step_dtype),
        )
        self.trajectory_regressor = nn.Linear(config.embedding_dim, 1)
        self.post_init()

    def forward(
            self,
            discharge_capacity: torch.Tensor,
            first_discharge_capacity: torch.Tensor,
            state_of_health: torch.Tensor,
            time: torch.Tensor,
            voltage_per_time: torch.Tensor,
            current_per_time: torch.Tensor,
            mask: Optional[torch.Tensor]=None,
            label_discharge_capacity: Optional[torch.Tensor]=None,
            label_state_of_health: Optional[torch.Tensor]=None,
            output_attentions: Optional[bool]=None,
        ) -> RoBaTOutput:
        output = self.model(
            discharge_capacity=discharge_capacity,
            first_discharge_capacity=first_discharge_capacity,
            state_of_health=state_of_health,
            time=time,
            voltage_per_time=voltage_per_time,
            current_per_time=current_per_time,
            mask=mask,
            label_discharge_capacity=label_discharge_capacity,
            label_state_of_health=label_state_of_health,
            output_attentions=output_attentions
        )
        x0 = output.last_hidden_state[:, -1, :] # (batch_size, embedding_dim)
        x = self.ode_integral(
            x0=x0,
            t=self.ode_step.to(x0.device)
        ) # (ode_prediction_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)
        # (batch_size, ode_prediction_len, embedding_dim)

        ret_state_of_health = self.trajectory_regressor(x) # (batch_size, ode_prediction_len, 1)
        ret_state_of_health = ret_state_of_health.squeeze(2)
        # (batch_size, ode_prediction_len)

        ret_discharge_capacity = ret_state_of_health * first_discharge_capacity.unsqueeze(1) # (batch_size, ode_prediction_len)

        label_mask = label_discharge_capacity == self.config.padding_value
        loss = self._calc_loss(
            ret_discharge_capacity=ret_discharge_capacity,
            label_discharge_capacity=label_discharge_capacity,
            ret_state_of_health=ret_state_of_health,
            label_state_of_health=label_state_of_health,
            mask=label_mask
        )

        return RoBaTOutput(
            loss=loss,
            discharge_capacity=ret_discharge_capacity,
            state_of_health=ret_state_of_health,
            multi_time_attn_weights=output.multi_time_attn_weights,
            time_level_rotary_attn_weights=output.time_level_rotary_attn_weights,
            cycle_level_rotary_attn_weights=output.cycle_level_rotary_attn_weights,
        )
# %%