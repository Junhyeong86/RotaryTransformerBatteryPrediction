# %%
from collections import defaultdict
from typing import Literal, Optional, List, Dict

import numpy

from datasets import (
    DatasetDict,
    Dataset,
    load_dataset,
    Features,
    Sequence,
    Value,
    Array2D
)

RENAME_COLUMN_DICT = {
    "cycle_idx": "cycle_idx",
    "time": "time",
    "voltage_per_time": "voltage_per_time",
    "current_per_time": "current_per_time",
    "mask": "mask",
    "first_discharge_capacity": "first_discharge_capacity",
    "state_of_health_per_cycle": "state_of_health",
    "label_discharge_capacity_per_cycle": "label_discharge_capacity",
    "label_state_of_health_per_cycle": "label_state_of_health",
}

# %%
def random_time_sampling(
        batch: dict,
        random_time_len: int=128,
        sampling_count: int=64,
    ):
    """
    Randomly sample time data from the batch.
    """
    batch_ref = batch["ref"] # (cell_idx)
    batch_cycle_idx = batch["cycle_idx"] # (cell_idx, cycle_len)
    batch_discharge_capacity_per_cycle = batch["discharge_capacity_per_cycle"] # (cell_idx, cycle_len)
    batch_charge_time = batch["time"] # (cell_idx, cycle_len, time_len)
    batch_charge_voltage_per_time = batch["voltage_per_time"] # (cell_idx, cycle_len, time_len)
    batch_charge_current_per_time = batch["current_per_time"] # (cell_idx, cycle_len, time_len)

    ret = defaultdict(list)
    for _ in range(sampling_count):
        cycle_iterator = zip(
            batch_ref,
            batch_cycle_idx,
            batch_discharge_capacity_per_cycle,
            batch_charge_time,
            batch_charge_voltage_per_time,
            batch_charge_current_per_time,
        )
        for cycle_iter in cycle_iterator:
            ref, cycle_i, cycle_dq = cycle_iter[:3]
            time_iterator = zip(*cycle_iter[3:])
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

            ret["ref"].append(ref)
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
        padding_side: Literal["left", "right"]="left",
        for_trajectory_pred: Optional[bool]=False,
        trajectory_pred_len: int=128,
    ):
    """
    Split the cycle data into chunks of length `split_cycle_len`.
    """
    batch_ref = batch["ref"] # (cell_idx)
    batch_cycle_idx = batch["cycle_idx"] # (cell_idx, cycle_len)
    batch_discharge_capacity_per_cycle = batch["discharge_capacity_per_cycle"] # (cell_idx, cycle_len)
    batch_charge_time = batch["time"] # (cell_idx, cycle_len, time_len)
    batch_charge_voltage_per_time = batch["voltage_per_time"] # (cell_idx, cycle_len, time_len)
    batch_charge_current_per_time = batch["current_per_time"] # (cell_idx, cycle_len, time_len)

    ret = defaultdict(list)
    cycle_iterator = zip(
        batch_ref,
        batch_cycle_idx,
        batch_discharge_capacity_per_cycle,
        batch_charge_time,
        batch_charge_voltage_per_time,
        batch_charge_current_per_time,
    )
    for cycle_iter in cycle_iterator:
        (
            ref,
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

        time_len = cycle_t.shape[1]
        last_chunk_pos = (
            len(cycle_idx) - 1
            if not for_trajectory_pred # nex cycle from previous cycles
            else len(cycle_idx) - trajectory_pred_len
        )
        if last_chunk_pos <= 0:
            continue
        # if 1,000 cycle and SOH prediction, 999(1,000-1) is the last input cycle
        # if 1,000 cycle and trajectory prediction with 100 cycle,
        # 900(1,000-100) is the last input cycle
        chunks_idx = numpy.array_split(
            range(last_chunk_pos),
            numpy.arange(split_cycle_len, last_chunk_pos, split_cycle_len),
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
                chunk_label_dq[padding_len:] = cycle_dq[chunk_idx + 1] # next cycle from previous cycles
                chunk_cycle_soh[padding_len:] = cycle_dq[chunk_idx] / cycle_dq[chunk_idx[0]]
                chunk_label_soh[padding_len:] = cycle_dq[chunk_idx + 1] / cycle_dq[chunk_idx[0]] # next cycle from previous cycles
                chunk_t[padding_len:] = cycle_t[chunk_idx]
                chunk_v[padding_len:] = cycle_v[chunk_idx]
                chunk_i[padding_len:] = cycle_i[chunk_idx]
            elif padding_side == "right":
                raise ValueError("right padding is not supported")
            else:
                raise ValueError(f"Invalid padding_side: {padding_side}")

            if for_trajectory_pred:
                trajectory_prediction_idx = (
                    numpy.arange(trajectory_pred_len)
                    + chunk_idx[-1] + 1
                )
                chunk_label_dq = cycle_dq[
                    trajectory_prediction_idx
                ]
                chunk_label_soh = cycle_dq[
                    trajectory_prediction_idx
                ] / cycle_dq[chunk_idx[0]]

            ret["ref"].append(ref)
            ret["mask"].append(chunk_mask)
            ret["cycle_idx"].append(chunk_cycle_idx)
            ret["first_discharge_capacity"].append(cycle_dq[chunk_idx[0]])
            ret["state_of_health_per_cycle"].append(chunk_cycle_soh)
            ret["label_discharge_capacity_per_cycle"].append(chunk_label_dq)
            ret["label_state_of_health_per_cycle"].append(chunk_label_soh)
            ret["time"].append(chunk_t)
            ret["voltage_per_time"].append(chunk_v)
            ret["current_per_time"].append(chunk_i)
    return dict(ret)

# %%
def ds_preprocess_for_train(
        ds_name: str,
        model_purpose: Literal["next_cycle_prediction", "trajectory_prediction"],
        random_time_len: int,
        sampling_count: int,
        split_cycle_len: int,
        padding_value: float = -100.0,
        padding_side: Literal["left", "right"] = "left",
        trajectory_pred_len: Optional[int] = None,
    ) -> DatasetDict:
    ds = load_dataset(ds_name)

    # random time sampling
    ds = ds.map(
        random_time_sampling,
        batched=True,
        batch_size=8,
        remove_columns=ds["train"].column_names,
        fn_kwargs={
            "random_time_len": random_time_len,
            "sampling_count": sampling_count,
        },
        desc="Random time sampling",
    )
    features = Features({
        "ref": Value(dtype="string"),
        "cycle_idx": Sequence(Value(dtype="int32")),
        "discharge_capacity_per_cycle": Sequence(Value(dtype="float32")),
        "time": Array2D(shape=(None, random_time_len), dtype="float32"),
        "voltage_per_time": Array2D(shape=(None, random_time_len), dtype="float32"),
        "current_per_time": Array2D(shape=(None, random_time_len), dtype="float32"),
    })
    ds = ds.cast(features)

    # split cycle
    split_cycle_kwargs = {
        "split_cycle_len": split_cycle_len,
        "padding_value": padding_value,
        "padding_side": padding_side,
        "for_trajectory_pred": False
    }
    if model_purpose == "trajectory_prediction":
        split_cycle_kwargs.update({
            "for_trajectory_pred": True,
            "trajectory_pred_len": trajectory_pred_len,
        })
    ds = ds.map(
        split_cycle,
        batched=True,
        batch_size=8,
        remove_columns=ds["train"].column_names,
        fn_kwargs=split_cycle_kwargs,
        desc="Split cycle",
    )
    features = Features({
        "ref": Value(dtype="string"),
        "cycle_idx": Sequence(Value(dtype="int64")),
        "time": Array2D(shape=(split_cycle_len, random_time_len), dtype="float32"),
        "voltage_per_time": Array2D(shape=(split_cycle_len, random_time_len), dtype="float32"),
        "current_per_time": Array2D(shape=(split_cycle_len, random_time_len), dtype="float32"),
        "mask": Sequence(Value(dtype="bool")),
        "first_discharge_capacity": Value(dtype="float32"),
        "state_of_health_per_cycle": Sequence(Value(dtype="float32")),
        "label_discharge_capacity_per_cycle": Sequence(Value(dtype="float32")),
        "label_state_of_health_per_cycle": Sequence(Value(dtype="float32")),
    })
    ds = ds.cast(features)

    ds = ds.rename_columns(RENAME_COLUMN_DICT)
    ds = ds.select_columns(list(RENAME_COLUMN_DICT.values()))

    # set format
    ds.set_format("torch")
    return ds

# %%
def ds_preprocess_for_eval(
        ds_name: str,
        model_purpose: Literal["next_cycle_prediction", "trajectory_prediction"],
        random_time_len: int,
        split_cycle_len: int,
        eval_ds_name: Literal["validation", "test"]="test",
        padding_value: float = -100.0,
        padding_side: Literal["left", "right"] = "left",
        trajectory_pred_len: Optional[int] = None,
    ) -> DatasetDict:
    ds = load_dataset(ds_name)

    # random time sampling
    ds = ds.map(
        random_time_sampling,
        batched=True,
        batch_size=8,
        remove_columns=ds["train"].column_names,
        fn_kwargs={
            "random_time_len": random_time_len,
            "sampling_count": 1,  # for eval, we sample only once
        },
        desc="Random time sampling",
    )
    eval_ds = {}
    for i, eval_d in enumerate(ds[eval_ds_name]):
        eval_key = f"{eval_d['ref']}/{eval_ds_name}/{i}"
        eval_d = {key: [value] for key, value in eval_d.items()}
        eval_ds[eval_key] = Dataset.from_dict(eval_d)
    eval_ds = DatasetDict(eval_ds)
    features = Features({
        "ref": Value(dtype="string"),
        "cycle_idx": Sequence(Value(dtype="int32")),
        "discharge_capacity_per_cycle": Sequence(Value(dtype="float32")),
        "time": Array2D(shape=(None, random_time_len), dtype="float32"),
        "voltage_per_time": Array2D(shape=(None, random_time_len), dtype="float32"),
        "current_per_time": Array2D(shape=(None, random_time_len), dtype="float32"),
    })
    eval_ds = eval_ds.cast(features)

    # split cycle
    split_cycle_kwargs = {
        "split_cycle_len": split_cycle_len,
        "padding_value": padding_value,
        "padding_side": padding_side,
        "for_trajectory_pred": False
    }
    if model_purpose == "trajectory_prediction":
        split_cycle_kwargs.update({
            "for_trajectory_pred": True,
            "trajectory_pred_len": trajectory_pred_len,
        })
    eval_ds = eval_ds.map(
        split_cycle,
        batched=True,
        batch_size=8,
        remove_columns=eval_ds[eval_key].column_names,
        fn_kwargs=split_cycle_kwargs,
        desc="Split cycle",
    )
    features = Features({
        "ref": Value(dtype="string"),
        "cycle_idx": Sequence(Value(dtype="int64")),
        "time": Array2D(shape=(split_cycle_len, random_time_len), dtype="float32"),
        "voltage_per_time": Array2D(shape=(split_cycle_len, random_time_len), dtype="float32"),
        "current_per_time": Array2D(shape=(split_cycle_len, random_time_len), dtype="float32"),
        "mask": Sequence(Value(dtype="bool")),
        "first_discharge_capacity": Value(dtype="float32"),
        "state_of_health_per_cycle": Sequence(Value(dtype="float32")),
        "label_discharge_capacity_per_cycle": Sequence(Value(dtype="float32")),
        "label_state_of_health_per_cycle": Sequence(Value(dtype="float32")),
    })
    eval_ds = eval_ds.cast(features)

    eval_ds = eval_ds.rename_columns(RENAME_COLUMN_DICT)
    eval_ds = eval_ds.select_columns(list(RENAME_COLUMN_DICT.values()))

    # set format
    eval_ds.set_format("torch")
    return eval_ds

# %%