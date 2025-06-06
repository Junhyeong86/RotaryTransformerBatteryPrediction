# %%
from typing import List, Dict

import torch

# %%
def collate_fn(batch: List[Dict[str, torch.Tensor]]):
    first_discharge_capacity = torch.stack(
        [item["first_discharge_capacity"] for item in batch]
    )
    state_of_health = torch.stack(
        [item["state_of_health"] for item in batch]
    )
    time = torch.stack(
        [item["time"] for item in batch]
    )
    voltage_per_time = torch.stack(
        [item["voltage_per_time"] for item in batch]
    )
    current_per_time = torch.stack(
        [item["current_per_time"] for item in batch]
    )
    mask = torch.stack(
        [item["mask"] for item in batch]
    )
    label_discharge_capacity = None
    if "label_discharge_capacity" in batch[0].keys():
        label_discharge_capacity = torch.stack(
            [item["label_discharge_capacity"] for item in batch]
        )
    if "label_state_of_health" in batch[0].keys():
        label_state_of_health = torch.stack(
            [item["label_state_of_health"] for item in batch]
        )

    ret = {
        "first_discharge_capacity": first_discharge_capacity,
        "state_of_health": state_of_health,
        "time": time,
        "voltage_per_time": voltage_per_time,
        "current_per_time": current_per_time,
        "mask": mask,
        "label_discharge_capacity": label_discharge_capacity,
        "label_state_of_health": label_state_of_health,
    }
    return ret

# %%