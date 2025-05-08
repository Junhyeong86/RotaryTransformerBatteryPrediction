from typing import Optional

import torch

# %%
class RoBaTDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            discharge_capacity: torch.Tensor,
            first_discharge_capacity: torch.Tensor,
            state_of_health: torch.Tensor,
            time: torch.Tensor,
            voltage_per_time: torch.Tensor,
            current_per_time: torch.Tensor,
            mask: torch.Tensor,
            label_discharge_capacity: Optional[torch.Tensor]=None,
            label_state_of_health: Optional[torch.Tensor]=None,
        ):
        self.discharge_capacity = discharge_capacity
        self.first_discharge_capacity = first_discharge_capacity
        self.state_of_health = state_of_health
        self.time = time
        self.voltage_per_time = voltage_per_time
        self.current_per_time = current_per_time
        self.mask = mask
        self.label_discharge_capacity = label_discharge_capacity
        self.label_state_of_health = label_state_of_health

    def __len__(self):
        return len(self.discharge_capacity)

    def __getitem__(self, idx):
        ret = {
            "discharge_capacity": self.discharge_capacity[idx],
            "first_discharge_capacity": self.first_discharge_capacity[idx],
            "state_of_health": self.state_of_health[idx],
            "time": self.time[idx],
            "voltage_per_time": self.voltage_per_time[idx],
            "current_per_time": self.current_per_time[idx],
            "mask": self.mask[idx],
        }
        if self.label_discharge_capacity is not None:
            ret["label_discharge_capacity"] = self.label_discharge_capacity[idx]
        if self.label_state_of_health is not None:
            ret["label_state_of_health"] = self.label_state_of_health[idx]
        return ret