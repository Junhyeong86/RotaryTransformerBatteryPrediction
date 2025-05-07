# %%
import numpy

import torch
import torch.nn.functional as F

def compute_metrics(eval_preds):
    discharge_capacity = eval_preds.predictions[1]
    state_of_health = eval_preds.predictions[2]

    label_discharge_capacity = eval_preds.label_ids
    label_discharge_capacity = numpy.array(label_discharge_capacity)
    label_state_of_health = label_discharge_capacity / label_discharge_capacity[:, 0][:, None] # (batch_size, cycle_len)

    discharge_capacity_mse = F.mse_loss(
        torch.FloatTensor(discharge_capacity),
        torch.FloatTensor(label_discharge_capacity),
    ) * 1e3 # to mAh loss of discharge capacity
    discharge_capacity_mae = F.l1_loss(
        torch.FloatTensor(discharge_capacity),
        torch.FloatTensor(label_discharge_capacity),
    ) * 1e3 # to mAh loss of discharge capacity

    state_of_health_mse = F.mse_loss(
        torch.FloatTensor(state_of_health),
        torch.FloatTensor(label_state_of_health),
    ) * 100 # to percentage loss of state of health
    state_of_health_mae = F.l1_loss(
        torch.FloatTensor(state_of_health),
        torch.FloatTensor(label_state_of_health),
    ) * 100 # to percentage loss of state of health

    return {
        "discharge_capacity_mse": discharge_capacity_mse.item(),
        "discharge_capacity_mae": discharge_capacity_mae.item(),
        "state_of_health_mse": state_of_health_mse.item(),
        "state_of_health_mae": state_of_health_mae.item(),
    }

# %%