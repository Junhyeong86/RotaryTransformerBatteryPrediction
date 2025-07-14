# %%
from collections import defaultdict
from typing import Literal, Optional, List, Dict
from logging import getLogger

import numpy
from scipy.interpolate import interp1d

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
    "cell_id": "cell_id",
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
def connect_discrete_time(
        batch: Dict[str, List],
        time_threshold: float = 1,
    ):
    """
    Connect the discrete time data into a single time series.
    """
    batch_time = batch["time"] # (cell_idx, cycle_len, time_len)

    ret = defaultdict(list)
    for cell_time in batch_time:
        ret_cell_time = []
        for time in cell_time:
            time = numpy.array(time, dtype=numpy.float32) # (time_len, )
            time_diff = numpy.diff(time)
            large_time_diff_indices = numpy.where(
                numpy.abs(time_diff) > time_threshold
            )[0]
            if len(large_time_diff_indices) == 0:
                ret_cell_time.append(time)
                continue
            time_to_be_added = numpy.zeros_like(time, dtype=numpy.float32)
            for i in large_time_diff_indices:
                time_to_be_added[i+1:] += -time_diff[i] + numpy.median(time_diff)
            ret_cell_time.append(time + time_to_be_added)
        ret["time"].append(ret_cell_time)
    return ret

# %%
def interpolate_discharge_capacity_per_cycle(
        batch: Dict[str, List],
        interpolate_soh_threshold: float = 0.005,
    ) -> Dict[str, List]:
    batch_cycle_idx = batch["cycle_idx"] # (cell_idx, cycle_len)
    batch_discharge_capacity_per_cycle = batch["discharge_capacity_per_cycle"] # (cell_idx, cycle_len)
    batch_time = batch["time"] # (cell_idx, cycle_len, time_len)
    batch_voltage_per_time = batch["voltage_per_time"] # (cell_idx, cycle_len, time_len)
    batch_current_per_time = batch["current_per_time"] # (cell_idx, cycle_len, time_len)

    ret: Dict[str, List] = defaultdict(list)
    cell_iterator = zip(
        batch_cycle_idx,
        batch_discharge_capacity_per_cycle,
        batch_time,
        batch_voltage_per_time,
        batch_current_per_time,
    )
    for cell_iter in cell_iterator:
        (
            cell_cycle_idx,
            cell_dq,
            cell_time, # (cycle_len, time_len)
            cell_voltage, # (cycle_len, time_len)
            cell_current, # (cycle_len, time_len)
        ) = cell_iter
        cell_cycle_idx = numpy.array(cell_cycle_idx, dtype=numpy.float32) # (cycle_len, )
        cell_dq = numpy.array(cell_dq, dtype=numpy.float32) # (cycle_len, )
        cycle_soh = cell_dq / cell_dq[0]  # (cycle_len, )
        interped_mask = numpy.zeros_like(cell_dq, dtype=bool)
        last_contiguous_mask = numpy.zeros_like(cell_dq, dtype=bool)
        while True:
            masked_pos = numpy.where(
                numpy.abs(numpy.diff(cycle_soh[~last_contiguous_mask])) >= interpolate_soh_threshold
            )[0] + 1 # change next position when difference is larger than threshold
            if len(masked_pos) == 0:
                break
            interped_mask[masked_pos] = True
            last_contiguous_mask = (
                (interped_mask[::-1].cumprod() == True) # reverse cumulative product to find last contiguous block
                [::-1]  # reverse back to original order
            )
            mask = interped_mask & (~last_contiguous_mask)  # only interpolate inner values
            cycle_soh[mask] = interp1d(
                cell_cycle_idx[~interped_mask],
                cycle_soh[~interped_mask],
            )(cell_cycle_idx[mask])
        cell_cycle_idx = cell_cycle_idx[~last_contiguous_mask]
        cell_dq = cell_dq[~last_contiguous_mask]
        if last_contiguous_mask.sum() > 0:
            cell_time = cell_time[:-last_contiguous_mask.sum()]
            cell_voltage = cell_voltage[:-last_contiguous_mask.sum()]
            cell_current = cell_current[:-last_contiguous_mask.sum()]
            interped_mask = interped_mask[:-last_contiguous_mask.sum()]

        # interpolate the columns
        cell_dq[interped_mask] = interp1d(
            cell_cycle_idx[~interped_mask],
            cell_dq[~interped_mask],
        )(cell_cycle_idx[interped_mask])

        # store the results
        ret["cycle_idx"].append(cell_cycle_idx)
        ret["discharge_capacity_per_cycle"].append(cell_dq)
        ret["time"].append(cell_time)
        ret["voltage_per_time"].append(cell_voltage)
        ret["current_per_time"].append(cell_current)
        assert len(cell_dq) == len(cell_time) == len(cell_voltage) == len(cell_current), \
            f"len(cell_dq) {len(cell_dq)} != len(cell_time) {len(cell_time)} != len(cell_voltage) {len(cell_voltage)} != len(cell_current) {len(cell_current)}"
    assert len(ret["discharge_capacity_per_cycle"]) == len(batch_cycle_idx), f"len(ret['discharge_capacity_per_cycle']) {len(ret['discharge_capacity_per_cycle'])} != len(batch_cycle_idx) {len(batch_cycle_idx)}"
    return ret

# %%
def random_time_sampling(
        batch: Dict[str, List],
        random_time_len: int=128,
        sampling_count: int=64,
    ):
    """
    Randomly sample time data from the batch.
    """
    logger = getLogger(__name__ + ".random_time_sampling")
    batch_ref = batch["ref"] # (cell_idx, )
    batch_cell_id = batch["cell_id"] # (cell_idx, )
    batch_cycle_idx = batch["cycle_idx"] # (cell_idx, cycle_len)
    batch_discharge_capacity_per_cycle = batch["discharge_capacity_per_cycle"] # (cell_idx, cycle_len)
    batch_time = batch["time"] # (cell_idx, cycle_len, time_len)
    batch_voltage_per_time = batch["voltage_per_time"] # (cell_idx, cycle_len, time_len)
    batch_current_per_time = batch["current_per_time"] # (cell_idx, cycle_len, time_len)

    ret = defaultdict(list)
    for _ in range(sampling_count):
        cell_iterator = zip(
            batch_ref,
            batch_cell_id,
            batch_cycle_idx,
            batch_discharge_capacity_per_cycle,
            batch_time,
            batch_voltage_per_time,
            batch_current_per_time,
        )
        for cell_iter in cell_iterator:
            (
                cell_ref,
                cell_id,
                cell_cycle_idx,
                cell_dq,
                cell_time,
                cell_voltage,
                cell_current,
            ) = cell_iter

            time_len_per_cycle = numpy.array([
                len(time_t) for time_t in cell_iter[-3]
            ])
            cell_should_skip = (
                (time_len_per_cycle < 2 * random_time_len).sum()
                > time_len_per_cycle.shape[0] * 0.01
            )  # if more than 1% of cycles have too short time length
            if cell_should_skip:
                logger.warning(
                    f"Cell {cell_id} has too short time length time_len_per_cycle.mean() {time_len_per_cycle.mean()} < 2 * random_time_len {2 * random_time_len}. "
                    "skipping this cell."
                )
            cycle_iterator = zip(
                cell_time,
                cell_voltage,
                cell_current,
            )
            cycle_len = len(cell_cycle_idx)
            time = numpy.zeros((cycle_len, random_time_len), dtype=numpy.float32)
            voltage_per_time = numpy.zeros((cycle_len, random_time_len), dtype=numpy.float32)
            current_per_time = numpy.zeros((cycle_len, random_time_len), dtype=numpy.float32)
            cell_cycle_idx = numpy.array(cell_cycle_idx, dtype=numpy.float32)
            cell_dq = numpy.array(cell_dq, dtype=numpy.float32)

            for cycle_iter_idx, cycle_iter in enumerate(cycle_iterator):
                time_t, time_v, time_i = cycle_iter
                # time_t: (time_len, ), time_v: (time_len, ), time_i: (time_len, )
                if len(time_t) < 2 * random_time_len:
                    time_t = cell_time[cycle_iter_idx - 1]  # use previous cycle data
                    time_v = cell_voltage[cycle_iter_idx - 1]
                    time_i = cell_current[cycle_iter_idx - 1]
                    logger.info(
                        f"Cell {cell_id} cycle idx {cycle_iter_idx} has too short time length"
                        f" Changed to previous cycle data."
                    )
                random_idx = numpy.random.choice(
                    len(time_t),
                    size=random_time_len,
                    replace=False,
                )
                random_idx.sort()

                time[cycle_iter_idx] = numpy.array([time_t[i] for i in random_idx], dtype=numpy.float32)
                voltage_per_time[cycle_iter_idx] = numpy.array([time_v[i] for i in random_idx], dtype=numpy.float32)
                current_per_time[cycle_iter_idx] = numpy.array([time_i[i] for i in random_idx], dtype=numpy.float32)

            ret["ref"].append(cell_ref)
            ret["cell_id"].append(cell_id)
            ret["cycle_idx"].append(cell_cycle_idx)
            ret["discharge_capacity_per_cycle"].append(cell_dq)
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
    batch_ref = batch["ref"] # (cell_idx, )
    batch_cell_id = batch["cell_id"] # (cell_idx)
    batch_cycle_idx = batch["cycle_idx"] # (cell_idx, cycle_len)
    batch_discharge_capacity_per_cycle = batch["discharge_capacity_per_cycle"] # (cell_idx, cycle_len)
    batch_time = batch["time"] # (cell_idx, cycle_len, time_len)
    batch_voltage_per_time = batch["voltage_per_time"] # (cell_idx, cycle_len, time_len)
    batch_current_per_time = batch["current_per_time"] # (cell_idx, cycle_len, time_len)

    ret = defaultdict(list)
    cell_iterator = zip(
        batch_ref,
        batch_cell_id,
        batch_cycle_idx,
        batch_discharge_capacity_per_cycle,
        batch_time,
        batch_voltage_per_time,
        batch_current_per_time,
    )
    for cell_iter in cell_iterator:
        (
            cell_ref,
            cell_id,
            cell_cycle_idx,
            cell_dq,
            cell_time,
            cell_voltage,
            cell_current,
        ) = cell_iter

        cell_cycle_idx = numpy.array(cell_cycle_idx, dtype=numpy.float32) # (cycle_len, )
        cell_dq = numpy.array(cell_dq, dtype=numpy.float32) # (cycle_len, )
        cell_time = numpy.array(cell_time, dtype=numpy.float32) # (cycle_len, time_len)
        cell_voltage = numpy.array(cell_voltage, dtype=numpy.float32) # (cycle_len, time_len)
        cell_current = numpy.array(cell_current, dtype=numpy.float32) # (cycle_len, time_len)

        time_len = cell_time.shape[1]
        last_chunk_pos = (
            len(cell_cycle_idx) - 1
            if not for_trajectory_pred # nex cycle from previous cycles
            else len(cell_cycle_idx) - trajectory_pred_len
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
            chunk_time = numpy.full(
                (split_cycle_len, time_len),
                fill_value=padding_value,
                dtype=numpy.float32,
            )
            chunk_voltage = numpy.full(
                (split_cycle_len, time_len),
                fill_value=padding_value,
                dtype=numpy.float32,
            )
            chunk_current = numpy.full(
                (split_cycle_len, time_len),
                fill_value=padding_value,
                dtype=numpy.float32,
            )

            if padding_side == "left":
                padding_len = split_cycle_len - len(chunk_idx)
                chunk_mask[:padding_len] = True
                chunk_cycle_idx[padding_len:] = cell_cycle_idx[chunk_idx]
                chunk_label_dq[padding_len:] = cell_dq[chunk_idx + 1] # next cycle from previous cycles
                chunk_cycle_soh[padding_len:] = cell_dq[chunk_idx] / cell_dq[chunk_idx[0]]
                chunk_label_soh[padding_len:] = cell_dq[chunk_idx + 1] / cell_dq[chunk_idx[0]] # next cycle from previous cycles
                chunk_time[padding_len:] = cell_time[chunk_idx]
                chunk_voltage[padding_len:] = cell_voltage[chunk_idx]
                chunk_current[padding_len:] = cell_current[chunk_idx]
            elif padding_side == "right":
                raise ValueError("right padding is not supported")
            else:
                raise ValueError(f"Invalid padding_side: {padding_side}")

            if for_trajectory_pred:
                trajectory_prediction_idx = (
                    numpy.arange(trajectory_pred_len)
                    + chunk_idx[-1] + 1
                )
                chunk_label_dq = cell_dq[
                    trajectory_prediction_idx
                ]
                chunk_label_soh = cell_dq[
                    trajectory_prediction_idx
                ] / cell_dq[chunk_idx[0]]

            ret["ref"].append(cell_ref)
            ret["cell_id"].append(cell_id)
            ret["mask"].append(chunk_mask)
            ret["cycle_idx"].append(chunk_cycle_idx)
            ret["first_discharge_capacity"].append(cell_dq[chunk_idx[0]])
            ret["state_of_health_per_cycle"].append(chunk_cycle_soh)
            ret["label_discharge_capacity_per_cycle"].append(chunk_label_dq)
            ret["label_state_of_health_per_cycle"].append(chunk_label_soh)
            ret["time"].append(chunk_time)
            ret["voltage_per_time"].append(chunk_voltage)
            ret["current_per_time"].append(chunk_current)
    return dict(ret)

# %%
def ds_preprocess_for_train(
        ds_name: str,
        model_purpose: Literal["next_cycle_prediction", "trajectory_prediction"],
        random_time_len: int,
        sampling_count: int,
        split_cycle_len: int,
        connect_time_threshold: Optional[float] = 1.0,
        interpolate_soh_threshold: Optional[float] = 0.005, # None means no interpolation
        padding_value: float = -100.0,
        padding_side: Literal["left", "right"] = "left",
        trajectory_pred_len: Optional[int] = None,
        refs_to_remove: Optional[List[str]] = None,
    ) -> DatasetDict:
    ds = load_dataset(ds_name)
    # connect discrete time
    if connect_time_threshold is not None:
        ds = ds.map(
            connect_discrete_time,
            batched=True,
            batch_size=8,
            fn_kwargs={
                "time_threshold": connect_time_threshold,
            },
            desc="Connect discrete time",
        )
    # interpolate discharge capacity per cycle
    if interpolate_soh_threshold is not None:
        # interpolate cycle
        ds = ds.map(
            interpolate_discharge_capacity_per_cycle,
            batched=True,
            batch_size=8,
            fn_kwargs={
                "interpolate_soh_threshold": interpolate_soh_threshold,
            },
            desc="Interpolate cycle",
        )

    # random time sampling
    for key in ds.keys():
        if key == "train":
            fn_kwargs = {
                "random_time_len": random_time_len,
                "sampling_count": sampling_count,
            }
        else:
            # for eval, we sample only once
            fn_kwargs = {
                "random_time_len": random_time_len,
                "sampling_count": 1,
            }
        ds[key] = ds[key].map(
            random_time_sampling,
            batched=True,
            batch_size=8,
            remove_columns=ds[key].column_names,
            fn_kwargs=fn_kwargs,
            desc="Random time sampling",
        )
    features = Features({
        "ref": Value(dtype="string"),
        "cell_id": Value(dtype="string"),
        "cycle_idx": Sequence(Value(dtype="float32")),
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
        "cell_id": Value(dtype="string"),
        "cycle_idx": Sequence(Value(dtype="float32")),
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
    ds.set_format("torch")

    # remove ref
    if refs_to_remove is not None:
        # filter by ref
        ds = ds.filter(
            lambda x: x["ref"] not in refs_to_remove,
            desc="Filter by ref",
        )

    # set format
    return ds

# %%
def ds_preprocess_for_eval(
        ds_name: str,
        model_purpose: Literal["next_cycle_prediction", "trajectory_prediction"],
        random_time_len: int,
        split_cycle_len: int,
        connect_time_threshold: Optional[float] = 1.0,
        interpolate_soh_threshold: Optional[float] = 0.005, # None means no interpolation
        eval_ds_column: Literal["validation", "test"]="test",
        padding_value: float = -100.0,
        padding_side: Literal["left", "right"] = "left",
        trajectory_pred_len: Optional[int] = None,
    ) -> DatasetDict:
    if model_purpose == "trajectory_prediction" and trajectory_pred_len is None:
        raise ValueError("trajectory_pred_len must be specified for trajectory prediction model purpose")
    ds = load_dataset(ds_name)
    ds.pop("train") # remove train dataset
    # connect discrete time
    if connect_time_threshold is not None:
        ds = ds.map(
            connect_discrete_time,
            batched=True,
            batch_size=8,
            fn_kwargs={
                "time_threshold": connect_time_threshold,
            },
            desc="Connect discrete time",
        )
    # interpolate discharge capacity per cycle
    if interpolate_soh_threshold is not None:
        # interpolate cycle
        ds = ds.map(
            interpolate_discharge_capacity_per_cycle,
            batched=True,
            batch_size=8,
            fn_kwargs={
                "interpolate_soh_threshold": interpolate_soh_threshold,
            },
            desc="Interpolate cycle",
        )

    # random time sampling
    ds = ds.map(
        random_time_sampling,
        batched=True,
        batch_size=8,
        remove_columns=ds[eval_ds_column].column_names,
        fn_kwargs={
            "random_time_len": random_time_len,
            "sampling_count": 1,  # for eval, we sample only once
        },
        desc="Random time sampling",
    )
    features = Features({
        "ref": Value(dtype="string"),
        "cell_id": Value(dtype="string"),
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
        remove_columns=ds[eval_ds_column].column_names,
        fn_kwargs=split_cycle_kwargs,
        desc="Split cycle",
    )
    features = Features({
        "ref": Value(dtype="string"),
        "cell_id": Value(dtype="string"),
        "cycle_idx": Sequence(Value(dtype="float32")),
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

    # create eval dataset
    eval_ds = defaultdict(lambda : defaultdict(list))
    for i, eval_d in enumerate(ds[eval_ds_column]):
        eval_key = f"{eval_d['ref']}/{eval_d['cell_id']}"
        eval_ds[eval_key] = {key: eval_ds[eval_key][key] + [value] for key, value in eval_d.items()}
    eval_ds = {key: Dataset.from_dict(value) for key, value in eval_ds.items()}
    eval_ds = DatasetDict(eval_ds)

    # set format
    eval_ds.set_format("torch")
    return eval_ds

# %%