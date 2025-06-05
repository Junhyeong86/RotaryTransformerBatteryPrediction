# %%
import os
import time
from datetime import timedelta
from typing import Tuple, List

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from transformers import (
    TrainingArguments,
    Trainer,
)
from transformers.utils.logging import get_logger as transformers_get_logger
from safetensors.torch import load_model

from torch import nn

from torchinfo import summary

from robat.globals import MODELS_CONFIG_PATH
from robat.logger import get_logger
from robat.preprocess import collate_fn
from robat.compute_metrics import compute_metrics
from robat.utils.training_postprocess import get_eval_df, is_compatable_model_config
from robat.utils.transformer_trainer_utils import get_model_ckpt

def get_best_model_ckpt(
        cfg: DictConfig,
    ) -> str:
    """
    Get the best model checkpoint based on the evaluation results.
    """
    cfg_from_pretrained = cfg.model.from_pretrained
    eval_df = get_eval_df(cfg_from_pretrained.model_search_dir)
    assert not eval_df.empty, "No evaluation results found in the specified directory."
    ode_layer_check = (
        (cfg.model.model_purpose == "trajectory_prediction") &
        (cfg.model.config.loss_column == "discharge_capacity")
    )
    mask = eval_df["model.config"].apply(
        lambda x: is_compatable_model_config(x, cfg.model.config, ode_layer_check=ode_layer_check)
    )
    eval_df = eval_df[mask].reset_index(drop=True)
    if eval_df.empty:
        raise ValueError("No best model found for the given configuration.")

    if cfg_from_pretrained.greater_is_better:
        best_model_idx = eval_df[f"eval_{cfg_from_pretrained.best_criterion}"].idxmax()
    else:
        best_model_idx = eval_df[f"eval_{cfg_from_pretrained.best_criterion}"].idxmin()
    best_model_dirpath = eval_df.loc[best_model_idx, "dirpath"]
    best_model_ckpt = get_model_ckpt(best_model_dirpath)
    return best_model_ckpt/"model.safetensors"

# %%
@hydra.main(version_base=None, config_path=str(MODELS_CONFIG_PATH), config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main function to run the training script.

    Args:
        cfg (DictConfig): Configuration object containing the training parameters.
    """

    assert cfg.job_name == "train", "This script is only for training. Please use the correct job name."

    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Log the configuration
    OmegaConf.save(cfg, f"{hydra_output_dir}/config.yaml")

    # get the best model checkpoint if needed
    # to avoid loading the best model during repeat training, call best_model outside the for loop
    cfg_from_pretrained = cfg.model.from_pretrained
    if cfg_from_pretrained.from_pretrained and cfg_from_pretrained.pretrained_model_name_or_path == "best":
        best_model_path = get_best_model_ckpt(cfg)

    for i in range(cfg.get("repeat", 1)):
        training_output_dir = os.path.join(hydra_output_dir, f"repeat={i+1}")

        # call the logger
        logger = get_logger(
            level="INFO",
            file_name=f"{hydra_output_dir}/log.log",
            using_file_handler=True,
            using_stream_handler=True,
            root_logger=True,
        )
        logger.info("Starting training script...")
        logger.info(f"training_output_dir: {training_output_dir}")
        logger.info(OmegaConf.to_yaml(cfg))

        transformers_logger = transformers_get_logger()
        transformers_logger.propagate = True # to logging to the root logger

        # initialize the dataset
        ds = instantiate(cfg.training.ds)
        logger.info(f"Dataset loaded")
        logger.info(str(ds))

        # initialize the model
        model = instantiate(cfg.model)
        logger.info(f"Model loaded")
        if cfg_from_pretrained.from_pretrained and cfg_from_pretrained.pretrained_model_name_or_path == "best":
            logger.info("Loading the best model based on evaluation results...")
            missing, unexpected = load_model(model, best_model_path)
            logger.info(f"Best model loaded from: {best_model_path}")
            if missing:
                logger.warning(f"Missing parameters during model loading: {missing}")
            if unexpected:
                logger.warning(f"Unexpected parameters during model loading: {unexpected}")
        elif cfg_from_pretrained.from_pretrained and cfg_from_pretrained.pretrained_model_name_or_path != "best":
            logger.info(f"Loading model from pretrained path: {cfg_from_pretrained.pretrained_model_name_or_path}")
            model = model.from_pretrained(
                cfg_from_pretrained.pretrained_model_name_or_path,
                config=cfg.model.config,
            )

        # set training arguments
        training_args = dict(cfg.training.training_args)
        batch_size = training_args["per_device_train_batch_size"]
        num_training_steps = training_args.pop("num_train_steps")
        if num_training_steps > 0:
            training_args["num_train_epochs"] = num_training_steps / (len(ds["train"]) // batch_size + 1)
        training_args["output_dir"] = training_output_dir
        training_args = TrainingArguments(**training_args)
        logger.info(f"Training arguments: {training_args}")

        # Create the Trainer instance
        callbacks = instantiate(cfg.training.callbacks)
        callbacks = [cb for name, cb in callbacks.items()]
        trainer = Trainer(
            model=model,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            data_collator=collate_fn,
            args=training_args,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )

        # summarize the model
        device = next(model.parameters()).device
        input_data = collate_fn([ds["train"][i] for i in range(batch_size)])
        input_data = {k: v.to(device) for k, v in input_data.items() if "label" not in k}
        model_summary = summary(model, input_data=input_data, verbose=0)
        logger.info(f"Model summary:")
        logger.info(str(model_summary))

        # %%
        start_time = time.time()
        logger.info("Training started.")
        trainer.train()
        end_time = time.time()
        training_time = timedelta(seconds=end_time - start_time)
        logger.info(f"Training time: {training_time}")
        logger.info("Training completed")
        trainer.save_state()

# %%
if __name__ == "__main__":
    # Run the main function
    main()