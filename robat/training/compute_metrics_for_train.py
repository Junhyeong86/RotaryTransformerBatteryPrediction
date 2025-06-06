# %%
from logging import getLogger

import numpy

import torch
import torch.nn.functional as F

def masked_r2_score(y_true, y_pred, mask):
    """
    Compute the masked R^2 score.
    
    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.
        mask (torch.BoolTensor): Mask to apply to the tensors.
    
    Returns:
        float: The masked R^2 score.
    """
    y_true = y_true[~mask]
    y_pred = y_pred[~mask]

    residual_sum_of_squares = torch.sum((y_true - y_pred) ** 2)
    total_sum_of_squares = torch.sum((y_true - torch.mean(y_true)) ** 2)

    if total_sum_of_squares == 0 and residual_sum_of_squares == 0:
        return torch.FloatTensor(1.0, dtype=y_true.dtype, device=y_true.device)
    elif total_sum_of_squares == 0 and residual_sum_of_squares != 0:
        return torch.FloatTensor(0.0, dtype=y_true.dtype, device=y_true.device)
    else:
        return 1 - (residual_sum_of_squares / total_sum_of_squares)

def compute_metrics(eval_preds):
    logger = getLogger(__name__ + ".compute_metrics")
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

    r2 = masked_r2_score(
        label_discharge_capacity,
        discharge_capacity,
        mask,
    ) * 1e2 # to percentage R^2 score

    ret = {
        "discharge_capacity_mse": discharge_capacity_mse.item(),
        "discharge_capacity_mae": discharge_capacity_mae.item(),
        "state_of_health_mse": state_of_health_mse.item(),
        "state_of_health_mae": state_of_health_mae.item(),
        "r2": r2.item(),
    }

    for k, value in ret.items():
        logger.info(f"{k}: {value}")
    return ret

# %%