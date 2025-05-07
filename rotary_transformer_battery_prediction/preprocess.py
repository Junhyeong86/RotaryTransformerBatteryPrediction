# %%
from collections import defaultdict
from typing import Literal

import numpy
import torch

# %%
def random_time_sampling(
        batch: dict,
        random_time_len: int=128,
        sampling_count: int=64,
    ):
    """
    Randomly sample time data from the batch.

    Args:
        batch: dict
            - unfixed cycle length and unfixed time length
            - cycle_idx: (cell_idx, cycle_len)
            - discharge_capacity_per_cycle: (cell_idx, cycle_len)
            - time: (cell_idx, cycle_len, time_len)
            - voltage_per_time: (cell_idx, cycle_len, time_len)
            - current_per_time: (cell_idx, cycle_len, time_len)
        random_time_len: int
            - The length of the random time data to be sampled.
        sampling_count: int
            - The number of times to sample the data.
    Returns:
        ret: dict
            - unfixed cycle length and fixed time length
            - cycle_idx: (sampling_count * cell_idx, cycle_len)
            - discharge_capacity_per_cycle: (sampling_count * cell_idx, cycle_len, random_time_len)
            - time: (sampling_count * cell_idx, cycle_len, random_time_len)
            - voltage_per_time: (sampling_count * cell_idx, cycle_len, random_time_len)
            - current_per_time: (sampling_count * cell_idx, cycle_len, random_time_len)
    """
    batch_cycle_idx = batch["cycle_idx"] # (cell_idx, cycle_len)
    batch_discharge_capacity_per_cycle = batch["discharge_capacity_per_cycle"] # (cell_idx, cycle_len)
    batch_charge_time = batch["time"] # (cell_idx, cycle_len, time_len)
    batch_charge_voltage_per_time = batch["voltage_per_time"] # (cell_idx, cycle_len, time_len)
    batch_charge_current_per_time = batch["current_per_time"] # (cell_idx, cycle_len, time_len)

    ret = defaultdict(list)
    for _ in range(sampling_count):
        cycle_iterator = zip(
            batch_cycle_idx,
            batch_discharge_capacity_per_cycle,
            batch_charge_time,
            batch_charge_voltage_per_time,
            batch_charge_current_per_time,
        )
        for cycle_iter in cycle_iterator:
            time_iterator = zip(*cycle_iter[2:])
            cycle_i, cycle_dq = cycle_iter[:2]
            cycle_len = len(cycle_i)
            time = numpy.zeros((cycle_len, random_time_len), dtype=numpy.float32)
            voltage_per_time = numpy.zeros((cycle_len, random_time_len), dtype=numpy.float32)
            current_per_time = numpy.zeros((cycle_len, random_time_len), dtype=numpy.float32)
            cycle_i = numpy.array(cycle_i, dtype=numpy.int64)
            cycle_dq = numpy.array(cycle_dq, dtype=numpy.float32)

            for time_iter_idx, time_iter in enumerate(time_iterator):
                time_t, time_v, time_i = time_iter
                # time_dq: 0, time_t: (time_len, ), time_v: (time_len, ), time_i: (time_len, )
                random_idx = numpy.random.choice(
                    len(time_t),
                    size=random_time_len,
                    replace=False,
                )
                random_idx.sort()

                time[time_iter_idx] = numpy.array([time_t[i] for i in random_idx], dtype=numpy.float32)
                voltage_per_time[time_iter_idx] = numpy.array([time_v[i] for i in random_idx], dtype=numpy.float32)
                current_per_time[time_iter_idx] = numpy.array([time_i[i] for i in random_idx], dtype=numpy.float32)

            ret["cycle_idx"].append(cycle_i)
            ret["discharge_capacity_per_cycle"].append(cycle_dq)
            ret["time"].append(time)
            ret["voltage_per_time"].append(voltage_per_time)
            ret["current_per_time"].append(current_per_time)

    assert len(ret["cycle_idx"]) == sampling_count * len(batch_cycle_idx), f"len(ret['cycle_idx']) {len(ret['cycle_idx'])} != sampling_count {sampling_count} * cell_len {len(batch_cycle_idx)}"

    return dict(ret)

# %%
def split_cycle(
        batch,
        split_cycle_len: int=256,
        padding_value: float=-100.0,
        padding_side: Literal["left", "right"] = "left",
    ):
    """
    Split the cycle data into chunks of length `split_cycle_len`.
    Args:
        batch: dict
            - unfixed cycle length and fixed time length
            - cycle_idx: (cell_idx, cycle_len)
            - discharge_capacity_per_cycle: (cell_idx, cycle_len)
            - time: (cell_idx, cycle_len, time_len)
            - voltage_per_time: (cell_idx, cycle_len, time_len)
            - current_per_time: (cell_idx, cycle_len, time_len)
        split_cycle_len: int
            - The length of the split cycle data.
        padding_value: float
            - The value to use for padding.
        padding_side: str
            - The side to pad the data. Can be "left" or "right".
    Returns:
        ret: dict
            - fixed cycle length and fixed time length
            - mask: (cell_idx, split_cycle_len)
            - cycle_idx: (cell_idx, split_cycle_len)
            - discharge_capacity_per_cycle: (cell_idx, split_cycle_len)
            - label_discharge_capacity_per_cycle: (cell_idx, split_cycle_len)
            - time: (cell_idx, split_cycle_len, time_len)
            - voltage_per_time: (cell_idx, split_cycle_len, time_len)
            - current_per_time: (cell_idx, split_cycle_len, time_len)
    """
    batch_cycle_idx = batch["cycle_idx"] # (cell_idx, cycle_len)
    batch_discharge_capacity_per_cycle = batch["discharge_capacity_per_cycle"] # (cell_idx, cycle_len)
    batch_charge_time = batch["time"] # (cell_idx, cycle_len, time_len)
    batch_charge_voltage_per_time = batch["voltage_per_time"] # (cell_idx, cycle_len, time_len)
    batch_charge_current_per_time = batch["current_per_time"] # (cell_idx, cycle_len, time_len)

    ret = defaultdict(list)
    cycle_iterator = zip(
        batch_cycle_idx,
        batch_discharge_capacity_per_cycle,
        batch_charge_time,
        batch_charge_voltage_per_time,
        batch_charge_current_per_time,
    )
    for cycle_iter in cycle_iterator:
        (
            cycle_idx,
            cycle_dq,
            cycle_t,
            cycle_v,
            cycle_i,
        ) = cycle_iter

        cycle_idx = numpy.array(cycle_idx, dtype=numpy.int64)
        cycle_dq = numpy.array(cycle_dq, dtype=numpy.float32)
        cycle_t = numpy.array(cycle_t, dtype=numpy.float32)
        cycle_v = numpy.array(cycle_v, dtype=numpy.float32)
        cycle_i = numpy.array(cycle_i, dtype=numpy.float32)

        time_len = len(cycle_t[0])
        chunks_idx = numpy.array_split(
            range(len(cycle_idx)-1),
            numpy.arange(split_cycle_len, len(cycle_i), split_cycle_len),
        )
        for chunk_idx in chunks_idx:
            if len(chunk_idx) == 0:
                continue
            chunk_mask = numpy.zeros(
                (split_cycle_len, ),
                dtype=numpy.bool
            )
            chunk_cycle_idx = numpy.full(
                (split_cycle_len, ),
                fill_value=padding_value,
                dtype=numpy.int64,
            )
            chunk_cycle_dq = numpy.full(
                (split_cycle_len, ),
                fill_value=padding_value,
                dtype=numpy.float32,
            )
            chunk_label_dq = numpy.full(
                (split_cycle_len, ),
                fill_value=padding_value,
                dtype=numpy.float32,
            )
            chunk_cycle_soh = numpy.full(
                (split_cycle_len, ),
                fill_value=padding_value,
                dtype=numpy.float32,
            )
            chunk_label_soh = numpy.full(
                (split_cycle_len, ),
                fill_value=padding_value,
                dtype=numpy.float32,
            )
            chunk_t = numpy.full(
                (split_cycle_len, time_len),
                fill_value=padding_value,
                dtype=numpy.float32,
            )
            chunk_v = numpy.full(
                (split_cycle_len, time_len),
                fill_value=padding_value,
                dtype=numpy.float32,
            )
            chunk_i = numpy.full(
                (split_cycle_len, time_len),
                fill_value=padding_value,
                dtype=numpy.float32,
            )

            if padding_side == "left":
                padding_len = split_cycle_len - len(chunk_idx)
                chunk_mask[:padding_len] = True
                chunk_cycle_idx[padding_len:] = cycle_idx[chunk_idx]
                chunk_cycle_dq[padding_len:] = cycle_dq[chunk_idx]
                chunk_label_dq[padding_len:] = cycle_dq[chunk_idx + 1] # next cycle from previous cycles
                chunk_cycle_soh[padding_len:] = cycle_dq[chunk_idx] / cycle_dq[chunk_idx[0]]
                chunk_label_soh[padding_len:] = cycle_dq[chunk_idx + 1] / cycle_dq[chunk_idx[0]] # next cycle from previous cycles
                chunk_t[padding_len:] = cycle_t[chunk_idx]
                chunk_v[padding_len:] = cycle_v[chunk_idx]
                chunk_i[padding_len:] = cycle_i[chunk_idx]
            elif padding_side == "right":
                chunk_mask[-len(chunk_idx):] = True
                chunk_cycle_idx[:len(chunk_idx)] = cycle_idx[chunk_idx]
                chunk_cycle_dq[:len(chunk_idx)] = cycle_dq[chunk_idx]
                chunk_label_dq[:len(chunk_idx)] = cycle_dq[chunk_idx + 1] # next cycle from previous cycles
                chunk_cycle_soh[:len(chunk_idx)] = cycle_dq[chunk_idx] / cycle_dq[chunk_idx[0]]
                chunk_label_soh[:len(chunk_idx)] = cycle_dq[chunk_idx + 1] / cycle_dq[chunk_idx[0]] # next cycle from previous cycles
                chunk_t[:len(chunk_idx)] = cycle_t[chunk_idx]
                chunk_v[:len(chunk_idx)] = cycle_v[chunk_idx]
                chunk_i[:len(chunk_idx)] = cycle_i[chunk_idx]
            else:
                raise ValueError(f"Invalid padding_side: {padding_side}")

            ret["mask"].append(chunk_mask)
            ret["cycle_idx"].append(chunk_cycle_idx)
            ret["first_discharge_capacity"].append(cycle_dq[chunk_idx[0]])
            ret["discharge_capacity_per_cycle"].append(chunk_cycle_dq)
            ret["state_of_health_per_cycle"].append(chunk_cycle_soh)
            ret["label_discharge_capacity_per_cycle"].append(chunk_label_dq)
            ret["label_state_of_health_per_cycle"].append(chunk_label_soh)
            ret["time"].append(chunk_t)
            ret["voltage_per_time"].append(chunk_v)
            ret["current_per_time"].append(chunk_i)
    return dict(ret)

# %%
def collate_fn(batch):
    discharge_capacity = torch.FloatTensor([
        item["discharge_capacity"] for item in batch
    ])
    first_discharge_capacity = torch.FloatTensor([
        item["first_discharge_capacity"] for item in batch
    ])
    state_of_health = torch.FloatTensor([
        item["state_of_health"] for item in batch
    ])
    time = torch.FloatTensor([
        item["time"] for item in batch
    ])
    voltage_per_time = torch.FloatTensor([
        item["voltage_per_time"] for item in batch
    ])
    current_per_time = torch.FloatTensor([
        item["current_per_time"] for item in batch
    ])
    mask = torch.BoolTensor([
        item["mask"] for item in batch
    ])

    label_discharge_capacity = None
    if "label_discharge_capacity" in batch[0]:
        label_discharge_capacity = torch.FloatTensor([
            item["label_discharge_capacity"] for item in batch
        ])
    if "label_state_of_health" in batch[0]:
        label_state_of_health = torch.FloatTensor([
            item["label_state_of_health"] for item in batch
        ])

    ret = {
        "discharge_capacity": discharge_capacity,
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