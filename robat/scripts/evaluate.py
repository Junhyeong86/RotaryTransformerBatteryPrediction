# %%
import os
import re
import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from tqdm.autonotebook import tqdm
from typing import (
    Literal,
    List,
    Dict,
    get_args,
)

import numpy
import pandas

from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib to accelerate the script

from matplotlib import pyplot

# %%
AVAILABLE_DS_NAMES = Literal[
    "Junhyeong86/battery-health-prediction",
]
AVAILABLE_DS_COLUMNS = Literal[
    "validation",
    "test",
]

# %%
def plot_label_vs_pred(
        ax: pyplot.Axes,
        ref_label_and_pred: Dict[str, List[numpy.ndarray]],
        column: str,
    ):
    # Save results
    get_xrange = lambda x: (
        numpy.floor(x.min() * 10) / 10,
        numpy.ceil(x.max() * 10) / 10
    )
    ref_label_and_pred = numpy.hstack([
        ref_label_and_pred[f"label_{column}"].reshape(-1, 1),
        ref_label_and_pred[f"pred_{column}"].reshape(-1, 1),
    ])
    kde = gaussian_kde(ref_label_and_pred.T)
    ax.scatter(
        *ref_label_and_pred.T,
        c=kde(ref_label_and_pred.T),
        s=1,
        alpha=0.5,
        cmap="viridis",
    )
    xrange = get_xrange(ref_label_and_pred[:, 0])
    yrange = get_xrange(ref_label_and_pred[:, 1])
    ax.plot(
        xrange,
        xrange,
        color="red",
        linestyle="--",
        label="y=x",
    )
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_title(f"label vs pred \n {column}")
    ax.set_xlabel("Ground truth (Ah)")
    ax.set_ylabel("Prediction (Ah)")
    return ax

# %%
def get_eval_ds_kwargs(
        ds_name: str,
        eval_ds_column: AVAILABLE_DS_COLUMNS,
        split_cycle_len: int,
        hydra_config: Dict,
        model_config: Dict
    ) -> Dict[str, str]:
    architecture = model_config["architectures"][0]
    training_config = hydra_config["training"]
    if architecture == "RoBaTForNextCyclePrediction":
        model_purpose = "next_cycle_prediction"
    elif architecture == "RoBaTForTrajectoryPrediction":
        model_purpose = "trajectory_prediction"
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}")
    return {
        "ds_name": ds_name,
        "model_purpose": model_purpose,
        "random_time_len": model_config["random_time_len"],
        "split_cycle_len": split_cycle_len,
        "connect_time_threshold": training_config["ds"]["connect_time_threshold"],
        "interpolate_soh_threshold": training_config["ds"]["interpolate_soh_threshold"],
        "eval_ds_column": eval_ds_column,
        "padding_value": model_config["padding_value"],
        "padding_side": model_config["padding_side"],
        "trajectory_pred_len": model_config["ode_prediction_len"],
    }

def evaluate(
        eval_root_dir: Path,
        ds_name: AVAILABLE_DS_NAMES="Junhyeong86/battery-health-prediction",
        split_cycle_len: int=256,
        save_fig: bool=True,
        save_eval_df: bool=True,
        log_filename: str=None,
        eval_ds_column: Literal["validation", "test"]="test",
    ) -> pandas.DataFrame:

    # the below imports are imported later to accelerate the script loading time
    import torch

    from accelerate import Accelerator

    from sklearn.metrics import (
        root_mean_squared_error,
        mean_absolute_error,
        r2_score,
    )

    from datasets import DatasetDict
    from omegaconf import OmegaConf

    from robat.logger import get_logger
    from robat.datasets.ds_preprocess import ds_preprocess_for_eval
    from robat import models
    from robat.training.trainer_base import SaveTopModelCallback

    logger = get_logger(
        level="INFO",
        file_name=log_filename,
        using_file_handler=True if not log_filename is None else False,
        using_stream_handler=True,
    )

    every_saved_models: Dict[Path, Path] = {}
    for root, _, files in os.walk(eval_root_dir):
        root = Path(root)
        if re.search(r"repeat=(\d+)", str(root.name)) is None:
            continue
        if any(p.startswith("_") for p in root.parts):
            continue  # skip hidden directories
        if not "trainer_state.json" in files:
            continue # skip directories if training is not terminated
        every_saved_models[root] = SaveTopModelCallback.get_model_ckpt(root)
    if not every_saved_models:
        raise ValueError(f"No saved models found in {eval_root_dir}")
    logger.info(f"Found {len(every_saved_models)} saved models in {eval_root_dir}")

    every_eval_ds: Dict[str, DatasetDict] = {} # pre-load datasets to avoid reloading
    for model_dir, model_ckpt in every_saved_models.items():
        ds_kwags = get_eval_ds_kwargs(
            ds_name=ds_name,
            eval_ds_column=eval_ds_column,
            split_cycle_len=split_cycle_len,
            hydra_config=OmegaConf.load(model_dir.parents[0] / "config.yaml"),
            model_config=json.load(open(model_ckpt / "config.json"))
        )
        eval_key = " ".join(f"{key}={value}" for key, value in ds_kwags.items())
        if eval_key in every_eval_ds:
            continue

        every_eval_ds[eval_key] = ds_preprocess_for_eval(**ds_kwags)
        logger.info(f"evalulation dataset prepared for {eval_key}")

    eval_df = defaultdict(list)
    accelerator = Accelerator()
    for model_dir, model_ckpt in every_saved_models.items():
        config = json.load(open(model_ckpt / "config.json"))
        metric_filepath = model_dir / f"{eval_ds_column}_metrics.json"
        ds_kwargs = get_eval_ds_kwargs(
            ds_name=ds_name,
            eval_ds_column=eval_ds_column,
            split_cycle_len=split_cycle_len,
            hydra_config=OmegaConf.load(model_dir.parents[0] / "config.yaml"),
            model_config=config,
        )
        eval_key = " ".join(f"{key}={value}" for key, value in ds_kwargs.items())
        architecture = config.pop("architectures")[0]

        if os.path.exists(metric_filepath):
            metrics = json.load(open(metric_filepath))
            logger.info(f"Metrics: {metrics}")
            logger.info(f"Metrics already exist for {model_dir}, skipping evaluation.")
        else:
            ds = every_eval_ds[eval_key]
            model = getattr(models, architecture).from_pretrained(
                model_ckpt,
                config=models.RoBaTConfig(**config),
            )
            model = accelerator.prepare(model)
            model.eval()
            logger.info(f"Loaded model on CPU from {model_ckpt}")
            logger.info(f"Model config: {config}")

            logger.info(f"Evaluation is performed on {accelerator.device}")
            progress_bar = tqdm(total=len(ds), desc=f"Evaluating {model_dir}")
            overall_label_and_pred = defaultdict(lambda: defaultdict(list))
            for cell_id, data in ds.items():
                ref = cell_id.split("/")[0]
                progress_bar.set_postfix({"processing cell id": cell_id})
                inp = {
                    "first_discharge_capacity": data["first_discharge_capacity"],
                    "state_of_health": data["state_of_health"],
                    "time": data["time"],
                    "voltage_per_time": data["voltage_per_time"],
                    "current_per_time": data["current_per_time"],
                    "mask": data["mask"],
                    "label_discharge_capacity": data["label_discharge_capacity"],
                    "label_state_of_health": data["label_state_of_health"],
                }
                inp = {k: v.to(accelerator.device) for k, v in inp.items()}
                with torch.no_grad():
                    pred = model(**inp)
                mask = data["label_discharge_capacity"] == config["padding_value"]
                pred_discharge_capacity = (
                    pred.discharge_capacity[~mask]
                    .detach()
                    .cpu()
                    .numpy()
                )
                label_discharge_capacity = (
                    data["label_discharge_capacity"][~mask]
                    .detach()
                    .cpu()
                    .numpy()
                )
                overall_label_and_pred[ref]["cell_id"].append(cell_id)
                overall_label_and_pred[ref]["pred_discharge_capacity"].append(pred_discharge_capacity)
                overall_label_and_pred[ref]["label_discharge_capacity"].append(label_discharge_capacity)
                progress_bar.update(1)
            progress_bar.close()

            for ref, ref_label_and_pred in overall_label_and_pred.items():
                for key, value in ref_label_and_pred.items():
                    overall_label_and_pred[ref][key] = numpy.hstack(value)

            # Save results
            if save_fig:
                fig, ax = pyplot.subplots(figsize=(8, 8))
                for ref, ref_label_and_pred in overall_label_and_pred.items():
                    fig_filepath = model_dir / f"{ref}_{eval_ds_column}_label vs pred discharge capacity.png"
                    logger.info(f"Saving label vs pred figure : {fig_filepath}")
                    ax = plot_label_vs_pred(
                        ax=ax,
                        ref_label_and_pred=ref_label_and_pred,
                        column="discharge_capacity",
                    )
                    fig.savefig(fig_filepath, dpi=100, bbox_inches="tight")
                    pyplot.close(fig)

            metrics: Dict[str, Dict[str, float]] = {}
            for ref, ref_label_and_pred in overall_label_and_pred.items():
                # Calculate metrics
                rmse = root_mean_squared_error(
                    y_true=ref_label_and_pred["label_discharge_capacity"],
                    y_pred=ref_label_and_pred["pred_discharge_capacity"]
                )
                mae = mean_absolute_error(
                    y_true=ref_label_and_pred["label_discharge_capacity"],
                    y_pred=ref_label_and_pred["pred_discharge_capacity"]
                )
                r2 = r2_score(
                    y_true=ref_label_and_pred["label_discharge_capacity"],
                    y_pred=ref_label_and_pred["pred_discharge_capacity"]
                )
                metrics[ref] ={
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2,
                }
            overall_rmse = root_mean_squared_error(
                y_true=numpy.hstack([v["label_discharge_capacity"] for v in overall_label_and_pred.values()]),
                y_pred=numpy.hstack([v["pred_discharge_capacity"] for v in overall_label_and_pred.values()])
            )
            overall_mae = mean_absolute_error(
                y_true=numpy.hstack([v["label_discharge_capacity"] for v in overall_label_and_pred.values()]),
                y_pred=numpy.hstack([v["pred_discharge_capacity"] for v in overall_label_and_pred.values()])
            )
            overall_r2 = r2_score(
                y_true=numpy.hstack([v["label_discharge_capacity"] for v in overall_label_and_pred.values()]),
                y_pred=numpy.hstack([v["pred_discharge_capacity"] for v in overall_label_and_pred.values()])
            )
            metrics["overall"] = {
                "RMSE": overall_rmse,
                "MAE": overall_mae,
                "R2": overall_r2,
            }
            with open(metric_filepath, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {metric_filepath}")
            logger.info(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")

        # Save evaluation dataframe
        eval_df["model_dir"].append(str(model_dir))
        for ref, ref_metrics in metrics.items():
            for key, value in ref_metrics.items():
                eval_df[f"{key}/{ref}"].append(value)
        for key, value in config.items():
            eval_df[key].append(value)

    eval_df = pandas.DataFrame(eval_df)
    if save_eval_df:
        eval_df.to_csv("evaluation_results.csv", index=False)
        logger.info(f"Evaluation results saved to evaluation_results.csv")
    logger.info("Evaluation completed")
    return eval_df


# %%
def main() -> None:
    parser = ArgumentParser(description="Evaluate saved models")
    parser.add_argument(
        "eval_root_dir",
        type=str,
        help="Root directory containing saved models for evaluation.",
    )
    parser.add_argument(
        "--ds_name",
        type=str,
        choices=get_args(AVAILABLE_DS_NAMES),
        default="Junhyeong86/battery-health-prediction",
        help="Name of the dataset to use for evaluation. Default: Junhyeong86/battery-health-prediction",
    )
    parser.add_argument(
        "--eval_ds_column",
        type=str,
        choices=get_args(AVAILABLE_DS_COLUMNS),
        default="test",
        help="Column of the dataset to use for evaluation. Default: test.",
    )
    parser.add_argument(
        "--split_cycle_len",
        type=int,
        default=256,
        help="Length of the cycle to split the dataset. Default: 256.",
    )
    parser.add_argument(
        "--no_save_fig",
        action="store_false",
        help="Disable to save the label vs prediction figures. Default: True.",
    )
    parser.add_argument(
        "--no_save_eval_df",
        action="store_false",
        help="Disable to save the evaluation results as a DataFrame. Default: True.",
    )
    parser.add_argument(
        "--log_filename",
        type=str,
        default=None,
        help="Name of the log file to save the evaluation logs. Default: evaluate.log.",
    )
    args = parser.parse_args()
    eval_root_dir = Path(args.eval_root_dir)
    ds_name = args.ds_name
    split_cycle_len = args.split_cycle_len
    save_fig = args.no_save_fig
    save_eval_df = args.no_save_eval_df
    log_filename = args.log_filename

    if not eval_root_dir.is_dir():
        raise ValueError(f"Provided eval_root_dir {eval_root_dir} is not a directory.")
    if ds_name not in get_args(AVAILABLE_DS_NAMES):
        raise ValueError(f"Provided ds_name {ds_name} is not in the available dataset names: {get_args(AVAILABLE_DS_NAMES)}")
    if not eval_root_dir.exists():
        raise ValueError(f"Provided eval_root_dir {eval_root_dir} does not exist.")

    evaluate(
        eval_root_dir=eval_root_dir,
        ds_name=ds_name,
        eval_ds_column=args.eval_ds_column,
        split_cycle_len=split_cycle_len,
        save_fig=save_fig,
        save_eval_df=save_eval_df,
        log_filename=log_filename,
    )

if __name__ == "__main__":
    main()