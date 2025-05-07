# %%
import numpy

import torch
import torch.nn.functional as F

def compute_metrics(eval_preds):
    discharge_capacity = eval_preds.predictions[0]
    state_of_health = eval_preds.predictions[1]

    label_discharge_capacity = eval_preds.label_ids[0]
    label_state_of_health = eval_preds.label_ids[1]
    label_discharge_capacity = numpy.array(label_discharge_capacity)

    mask = numpy.isclose(label_discharge_capacity, -100)
    valid_elements = numpy.sum(~mask)
    mask = torch.BoolTensor(mask)

    discharge_capacity = torch.FloatTensor(discharge_capacity).masked_fill(mask, -100)
    label_discharge_capacity = torch.FloatTensor(label_discharge_capacity).masked_fill(mask, -100)
    discharge_capacity_mse = F.mse_loss(
        discharge_capacity,
        label_discharge_capacity,
        reduction="sum",
    ) / valid_elements * 1e6 # to mAh loss of discharge capacity
    discharge_capacity_mae = F.l1_loss(
        discharge_capacity,
        label_discharge_capacity,
        reduction="sum",
    ) / valid_elements * 1e3 # to mAh loss of discharge capacity

    state_of_health = torch.FloatTensor(state_of_health).masked_fill(mask, -100)
    label_state_of_health = torch.FloatTensor(label_state_of_health).masked_fill(mask, -100)
    state_of_health_mse = F.mse_loss(
        state_of_health,
        label_state_of_health,
        reduction="sum",
    ) / valid_elements * 1e4 # to percentage loss of state of health
    state_of_health_mae = F.l1_loss(
        state_of_health,
        label_state_of_health,
        reduction="sum",
    ) / valid_elements * 1e2 # to percentage loss of state of health

    return {
        "discharge_capacity_mse": discharge_capacity_mse.item(),
        "discharge_capacity_mae": discharge_capacity_mae.item(),
        "state_of_health_mse": state_of_health_mse.item(),
        "state_of_health_mae": state_of_health_mae.item(),
    }

# %%