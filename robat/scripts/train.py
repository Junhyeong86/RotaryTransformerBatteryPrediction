# %%
import os
import time
from datetime import timedelta
from typing import Tuple, List

import hydra
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf

from torch import nn

from torchinfo import summary

from robat.globals import MODELS_CONFIG_PATH
from robat.logger import get_logger
from robat.training.compute_metrics_for_train import compute_metrics
from robat.training.postprocess_for_train import get_eval_df, is_compatable_model_config
from robat.training.trainer_base import (
    RoBaTTrainer,
    SaveTopModelCallback
)

# %%
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
        best_model_idx = eval_df[f"eval/{cfg_from_pretrained.best_criterion}"].idxmax()
    else:
        best_model_idx = eval_df[f"eval/{cfg_from_pretrained.best_criterion}"].idxmin()
    best_model_dirpath = eval_df.loc[best_model_idx, "dirpath"]

    best_model_ckpt = SaveTopModelCallback.get_model_ckpt(best_model_dirpath)
    return best_model_ckpt

# %%
@hydra.main(version_base=None, config_path=str(MODELS_CONFIG_PATH), config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main function to run the training script.

    Args:
        cfg (DictConfig): Configuration object containing the training parameters.
    """

    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Log the configuration
    OmegaConf.save(cfg, f"{hydra_output_dir}/config.yaml")

    # get the best model checkpoint if needed
    # to avoid loading the best model during repeat training, call best_model outside the for loop
    cfg_from_pretrained = cfg.model.from_pretrained
    if cfg_from_pretrained.from_pretrained and cfg_from_pretrained.pretrained_model_name_or_path == "best":
        pretrained_model_name_or_path = get_best_model_ckpt(cfg)
    elif cfg_from_pretrained.from_pretrained and cfg_from_pretrained.pretrained_model_name_or_path != "best":
        pretrained_model_name_or_path = cfg_from_pretrained.pretrained_model_name_or_path

    # call the logger
    logger = get_logger(
        level=cfg.logging.level,
        file_name=f"{hydra_output_dir}/log.log",
        using_file_handler=True,
        using_stream_handler=True,
    )
    logger.info(OmegaConf.to_yaml(cfg))

    # initialize the dataset
    ds = instantiate(cfg.training.ds)
    logger.info(f"Dataset loaded")
    logger.info(str(ds))

    for i in range(cfg.get("repeat", 1)):
        training_output_dir = os.path.join(hydra_output_dir, f"repeat={i+1}")

        logger.info("Starting training script...")
        logger.info(f"training_output_dir: {training_output_dir}")

        # initialize the model
        if not cfg_from_pretrained.from_pretrained:
            logger.info("Loading model from scratch")
            model = instantiate(cfg.model)
        else:
            logger.info(f"Loading model from pretrained path: {pretrained_model_name_or_path}")
            model_class = get_class(cfg.model._target_)
            model = model_class.from_pretrained(
                pretrained_model_name_or_path,
                config=instantiate(cfg.model.config),
            )
        logger.info(f"Model loaded")

        # set training arguments
        trainer_args = instantiate(cfg.training.trainer_args, output_dir=training_output_dir)
        logger.info(f"Trainer arguments: {trainer_args}")

        # set the optimizer and scheduler
        optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
        lr_scheduler = instantiate(cfg.training.lr_scheduler, optimizer=optimizer)

        # Create the Trainer instance
        callbacks = instantiate(cfg.training.callbacks)
        callbacks = [cb for name, cb in callbacks.items()]
        trainer = RoBaTTrainer(
            trainer_args=trainer_args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=ds["train"],
            eval_dataloader=ds["validation"],
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        # summarize the model
        logger.info(f"Model summary:")
        logger.info(str(trainer.summary()))

        # %%
        start_time = time.time()
        logger.info("Training started.")
        trainer.train()
        end_time = time.time()
        training_time = timedelta(seconds=end_time - start_time)
        logger.info(f"Training time: {training_time}")
        logger.info("Training completed")
        trainer.save_state(training_output_dir)
        logger.info(f"Training state saved to {training_output_dir}")

# %%
if __name__ == "__main__":
    # Run the main function
    main()