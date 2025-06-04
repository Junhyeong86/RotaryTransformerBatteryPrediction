# %%
import os
import time
from datetime import timedelta

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from transformers import (
    TrainingArguments,
    Trainer,
)

from torchinfo import summary

from robat.globals import MODELS_CONFIG_PATH
from robat.logger import get_logger
from robat.preprocess import collate_fn
from robat.compute_metrics import compute_metrics

# %%
@hydra.main(version_base=None, config_path=str(MODELS_CONFIG_PATH), config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main function to run the training script.

    Args:
        cfg (DictConfig): Configuration object containing the training parameters.
    """
    assert cfg.job_name == "train", "This script is only for training. Please use the correct job name."

    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.training_output_dir
    for _ in range(cfg.get("repeat", 1)):
        training_output_dir = os.path.join(hydra_output_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))

        # call the logger
        logger = get_logger(
            level="INFO",
            file_name=f"{training_output_dir}/log.log",
        )
        logger.info("Starting training script...")
        logger.info(f"training_output_dir: {training_output_dir}")
        logger.info(OmegaConf.to_yaml(cfg))

        # initialize the dataset
        ds = instantiate(cfg.training.ds)
        logger.info(f"Dataset loaded")
        logger.info(str(ds))

        # initialize the model
        model = instantiate(cfg.model)
        logger.info(f"Model loaded")

        # set training arguments
        training_args = dict(cfg.training.training_args)
        batch_size = training_args["per_device_train_batch_size"]
        num_training_steps = training_args.pop("num_train_steps")
        if num_training_steps > 0:
            training_args["num_train_epochs"] = num_training_steps / (len(ds["train"]) // batch_size + 1)
        training_args["training_output_dir"] = training_output_dir
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

        # Log the configuration
        OmegaConf.save(cfg, f"{training_output_dir}/config.yaml")

# %%
if __name__ == "__main__":
    # Run the main function
    main()