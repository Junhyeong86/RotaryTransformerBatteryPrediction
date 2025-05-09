# %%
import os
import re
import json
from typing import Optional
from collections import defaultdict
from argparse import ArgumentParser

import numpy
import pandas

# %%
def ckpt_exists(output_dir: str) -> bool:
    pattern = re.compile(
        r"/ckpt/\b\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\b" # yyyy-mm-dd_hh-mm-ss
    )
    if os.path.exists(output_dir):
        for root, dirs, _ in os.walk(output_dir):
            for directory in dirs:
                full_path = os.path.join(root, directory)
                if pattern.search(full_path):
                    return True
    return False

# %%
OUTPUT_DIR_FORMAT = os.path.join(
    "log",
    "pretraining",
    "time_transformer_num_layers={time_transformer_num_layers}"
    " cycle_transformer_num_layers={cycle_transformer_num_layers}"
    " learning_rate={learning_rate}"
    " lr_scheduler={lr_scheduler}"
    " loss_column={loss_column}"
    " training_dir_suffix={training_dir_suffix}",
)

# %%
# find pretrained model
def dirname2dict(dirname: str) -> dict:
    """
    Convert a directory name to a dictionary of parameters.
    """
    dir_pattern = re.compile(
        r"time_transformer_num_layers=(?P<time_transformer_num_layers>\d+)\s+"
        r"cycle_transformer_num_layers=(?P<cycle_transformer_num_layers>\d+)\s+"
        r"learning_rate=(?P<learning_rate>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s+"
        r"lr_scheduler=(?P<lr_scheduler>[\w-]+)\s+"
        r"loss_column=(?P<loss_column>(state_of_health|discharge_capacity))\s+"
        r"training_dir_suffix=(?P<training_dir_suffix>\d+)"
    )
    matched = dir_pattern.match(dirname)
    if matched:
        return {
            "time_transformer_num_layers": int(matched.group("time_transformer_num_layers")),
            "cycle_transformer_num_layers": int(matched.group("cycle_transformer_num_layers")),
            "learning_rate": float(matched.group("learning_rate")),
            "lr_scheduler": matched.group("lr_scheduler"),
            "loss_column": matched.group("loss_column"),
            "training_dir_suffix": int(matched.group("training_dir_suffix")),
        }
    else:
        return None

# %%
def get_eval_df(log_dir: Optional[str]=None) -> pandas.DataFrame:
    """
    Get the evaluation dataframe.
    """
    eval_df = defaultdict(list)
    log_dir = log_dir if log_dir else "log"
    for root, dirs, files in os.walk(log_dir):
        if os.path.basename(root).startswith("_"):
            # skip hidden directories
            continue
        for directory in dirs:
            dir_dict = dirname2dict(directory)
            if dir_dict is None:
                continue
            trainer_state = os.path.join(root, directory, "trainer_state.json")
            trainer_state = json.load(open(trainer_state))

            for log in trainer_state["log_history"]:
                if not "eval_loss" in log:
                    continue
                eval_df["dirpath"].append(os.path.join(root, directory))
                for key, value in dir_dict.items():
                    eval_df[key].append(value)
                for key, value in log.items():
                    eval_df[key].append(value)
    eval_df = pandas.DataFrame(eval_df)
    return eval_df

# %%
def get_model_ckpt(model_dirpath: str) -> str:
    """
    Get the model checkpoint from the directory.
    """
    pattern = re.compile(
        r"/ckpt/[^/]+/ckpt_epoch-\d+(\.\d+)?_[^/]+-\d+(\.\d+)?/checkpoint-\d+$"
    )
    _model_ckpt = []
    for root, dirs, files in os.walk(model_dirpath):
        for directory in dirs:
            full_path = os.path.join(root, directory)
            if pattern.search(full_path):
                _model_ckpt.append(full_path)

    if len(_model_ckpt) > 1:
        model_ckpt_str = "\n\t-" + "\n\t-".join(_model_ckpt)
        raise ValueError(f"Multiple model checkpoints found: {model_ckpt_str}")

    model_ckpt = _model_ckpt[0]
    return model_ckpt

# %%
if __name__ == "__main__":
    # %%
    parser = ArgumentParser(description="Train a model for SOH prediction.")
    parser.add_argument(
        "--time_transformer_num_layers",
        type=int,
        help="Number of layers in the time-level transformer.",
        required=True,
    )
    parser.add_argument(
        "--cycle_transformer_num_layers",
        type=int,
        help="Number of layers in the cycle-level transformer.",
        required=True,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate for the optimizer.",
        default=1e-5,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        help="Learning rate scheduler to use.",
        choices=["linear", "cosine"],
        default="cosine",
    )
    parser.add_argument(
        "--loss_column",
        type=str,
        help="Column name for the loss.",
    )
    parser.add_argument(
        "--training_dir_suffix",
        type=int,
        help="Suffix for the training directory.",
        default=0,
    )

    # %%
    # parsing arguments
    args = parser.parse_args()
    time_transformer_num_layers = args.time_transformer_num_layers
    cycle_transformer_num_layers = args.cycle_transformer_num_layers
    learning_rate = args.learning_rate
    lr_scheduler = args.lr_scheduler
    loss_column = args.loss_column
    training_dir_suffix = args.training_dir_suffix

    # fixed parameters
    ## parameters for dataset
    split_cycle_len = 256
    padding_value = -100.0
    padding_side = "left"
    random_time_len = 128
    sampling_count = 16
    ## parameters for model
    embedding_dim = 64
    feed_forward_hidden_dim = 128
    num_heads = 8
    ref_time_seq_len = 16
    time_transformer_output_dim = 4
    dropout = 0.0
    freq_base = 10_000.0
    rotary_embedding_dtype = "float32"
    max_sequence_len = 1024
    output_attentions = False
    ## parameters for training
    batch_size = 32
    num_training_steps = 20_000
    warmup_steps = 1_000
    weight_decay = 0.01
    input_datadict = {
        "discharge_capacity_per_cycle": "discharge_capacity",
        "first_discharge_capacity": "first_discharge_capacity",
        "state_of_health_per_cycle": "state_of_health",
        "time": "time",
        "voltage_per_time": "voltage_per_time",
        "current_per_time": "current_per_time",
        "mask": "mask",
        "label_discharge_capacity_per_cycle": "label_discharge_capacity",
        "label_state_of_health_per_cycle": "label_state_of_health",
    }

    # %%
    # output directory
    output_dir = OUTPUT_DIR_FORMAT.format(
        time_transformer_num_layers=time_transformer_num_layers,
        cycle_transformer_num_layers=cycle_transformer_num_layers,
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        loss_column=loss_column,
        training_dir_suffix=training_dir_suffix,
    )

    if ckpt_exists(output_dir):
        raise ValueError(f"checkpoint already exists at \n{output_dir}")

    # %%
    # get pretrained model for discharge_capacity loss
    if loss_column == "discharge_capacity":
        eval_df = get_eval_df(log_dir="log")
        mask = (
            (eval_df["time_transformer_num_layers"] == time_transformer_num_layers) &
            (eval_df["cycle_transformer_num_layers"] == cycle_transformer_num_layers)
        )
        eval_df = eval_df.loc[mask].reset_index(drop=True)
        if len(eval_df) == 0:
            raise ValueError(
                "checkpoint is required for discharge_capacity loss."
                "please pretrain the model with state_of_health loss with "
                f"time_transformer_num_layers={time_transformer_num_layers} "
                f"and cycle_transformer_num_layers={cycle_transformer_num_layers}"
            )
        best_model_dirpath = eval_df.loc[eval_df["eval_loss"].idxmin(), "dirpath"]
        best_model_ckpt = get_model_ckpt(best_model_dirpath)

    # %%
    import time
    from datetime import timedelta
    from logging import (
        Formatter,
        FileHandler,
        StreamHandler
    )

    import numpy
    import pandas

    import torch
    from torchinfo import summary

    from datasets import load_dataset, Features, Array2D, Sequence, Value
    from transformers import (
        Trainer,
        TrainingArguments
    )
    from transformers.utils import logging
    from transformers.trainer_callback import EarlyStoppingCallback

    from rotary_transformer_battery_prediction.models import (
        RoBaTConfig,
        RoBaTPreTrainedModel,
    )
    from rotary_transformer_battery_prediction.utils.transformer_trainer_utils import SaveTopModelsCallback
    from rotary_transformer_battery_prediction.preprocess import (
        split_cycle,
        random_time_sampling,
        collate_fn
    )
    from rotary_transformer_battery_prediction.compute_metrics import compute_metrics

    # %%
    # check if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # %%
    # logging
    logger = logging.get_logger()
    logger.setLevel(logging.INFO)

    formatter = Formatter("%(levelname)s %(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = FileHandler(os.path.join(output_dir, "training.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)

    logging.add_handler(file_handler)

    # %%
    # load dataset
    ds = load_dataset("Junhyeong86/battery-health-prediction")

    # %%
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
        "cycle_idx": Sequence(Value(dtype="int32")),
        "discharge_capacity_per_cycle": Sequence(Value(dtype="float32")),
        "time": Array2D(shape=(None, random_time_len), dtype="float32"),
        "voltage_per_time": Array2D(shape=(None, random_time_len), dtype="float32"),
        "current_per_time": Array2D(shape=(None, random_time_len), dtype="float32"),
    })
    ds = ds.cast(features)

    # %%
    # split cycle
    ds = ds.map(
        split_cycle,
        batched=True,
        batch_size=8,
        remove_columns=ds["train"].column_names,
        fn_kwargs={
            "split_cycle_len": split_cycle_len,
            "padding_value": padding_value,
            "padding_side": padding_side,
        },
        desc="Split cycle",
    )
    features = Features({
        "cycle_idx": Sequence(Value(dtype="int32")),
        "discharge_capacity_per_cycle": Sequence(Value(dtype="float32")),
        "time": Array2D(shape=(split_cycle_len, random_time_len), dtype="float32"),
        "voltage_per_time": Array2D(shape=(split_cycle_len, random_time_len), dtype="float32"),
        "current_per_time": Array2D(shape=(split_cycle_len, random_time_len), dtype="float32"),
        "mask": Sequence(Value(dtype="bool")),
        "label_discharge_capacity_per_cycle": Sequence(Value(dtype="float32")),
    })

    # %%
    # rename columns for dataset input
    ds = ds.rename_columns(input_datadict)
    ds = ds.select_columns(list(input_datadict.values()))

    # %%
    # shuffle dataset
    ds = ds.shuffle(seed=42)
    logger.info("Dataset preprocessing completed.")
    logger.info(f"random time length: {random_time_len}")
    logger.info(f"sampling count: {sampling_count}")
    logger.info(f"split cycle length: {split_cycle_len}")
    logger.info(f"padding value: {padding_value}")
    logger.info(f"padding side: {padding_side}")

    # %%
    # set format
    ds.set_format(type="torch")

    # %%
    # model
    if loss_column == "discharge_capacity":
        config = RoBaTConfig.from_pretrained(best_model_ckpt)
        model = RoBaTPreTrainedModel(best_model_ckpt, config=config)
    elif loss_column == "state_of_health":
        config = RoBaTConfig(
            embedding_dim=embedding_dim,
            feed_forward_hidden_dim=feed_forward_hidden_dim,
            num_heads=num_heads,
            time_transformer_num_layers=time_transformer_num_layers,
            ref_time_seq_len=ref_time_seq_len,
            time_transformer_output_dim=time_transformer_output_dim,
            cycle_transformer_num_layers=cycle_transformer_num_layers,
            dropout=dropout,
            freq_base=freq_base,
            loss_column=loss_column,
            rotary_embedding_dtype=rotary_embedding_dtype,
            max_sequence_len=max_sequence_len,
            output_attentions=output_attentions,
            padding_value=padding_value,
        )
        model = RoBaTPreTrainedModel(config=config)
    logger.info("Model config:")
    logger.info(str(config))

    # %%
    # training arguments
    num_train_epochs = num_training_steps / (len(ds["train"]) // batch_size + 1)

    training_args = {
        "output_dir": output_dir,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": num_train_epochs,
        "logging_steps": 1,
        "report_to": "tensorboard",
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "lr_scheduler_type": lr_scheduler,
        "optim": "adamw_torch",
        "learning_rate": learning_rate,
        "load_best_model_at_end": True,
        "metric_for_best_model": "loss",
    }
    training_args = TrainingArguments(**training_args)
    logger.info("Training arguments:")
    logger.info(str(training_args))

    # %%
    # callbacks
    save_top_models_callback = SaveTopModelsCallback(
        top_n=1,
        logger=logger,
    )
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
    callbacks = [
        save_top_models_callback,
        early_stopping_callback,
    ]

    # %%
    # summary
    trainer = Trainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collate_fn,
        args=training_args,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )
    device = next(model.parameters()).device
    input_data = collate_fn([ds["train"][i] for i in range(batch_size)])
    input_data = {k: v.to(device) for k, v in input_data.items() if "label" not in k}
    model_summarize = summary(model, input_data=input_data, verbose=0)
    logger.info("Model summary:")
    logger.info(str(model_summarize))

    # %%
    start_time = time.time()
    logger.info("Training started.")
    trainer.train()
    end_time = time.time()
    training_time = timedelta(seconds=end_time - start_time)
    logger.info(f"Training time: {training_time}")
    logger.info("Training completed.")
    trainer.save_state()

    # %%