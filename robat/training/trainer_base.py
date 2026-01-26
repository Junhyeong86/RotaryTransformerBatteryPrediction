# %%
# import libraries
import os
import re
import shutil
import json
import time
import inspect
from pathlib import Path
from datetime import timedelta, datetime
from collections import defaultdict
from tqdm.autonotebook import tqdm
from logging import getLogger
from dataclasses import dataclass, asdict
from typing import (
    Union,
    Literal,
    Optional,
    Dict,
    List,
    Callable
)

from datasets import Dataset
import numpy

import torch
from torchinfo import summary

from transformers import PreTrainedModel, SchedulerType

from accelerate import Accelerator

from ..models.battery_prediction_model import RoBaTOutput
from ..utils.json_encoder import JsonNumpyEncoder

# %%
MIXED_PRECISION_TYPE = Literal[
    "no",
    "fp16",
    "bf16",
    "fp8",
]
REPORT_TO_TYPE = Literal[
    "tensorboard",
    "wandb"
]


# %%
@dataclass
class RoBaTTrainerArguments:
    output_dir: str
    num_train_steps: int
    eval_steps: int
    logging_steps: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    mixed_precision: MIXED_PRECISION_TYPE = "fp16"
    report_to: REPORT_TO_TYPE = "wandb"
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False

# %%
@dataclass
class RoBaTTrainerState:
    global_step: int = 0
    epoch: float = 0.0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
    should_stop: bool = False
    should_save: bool = False

    def copy(self):
        return RoBaTTrainerState(**self.__dict__)


# %%
class RoBaTTrainerCallback:
    def __init__(self):
        """
        Initialize the callback.
        """
        self.logger = getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.propagate = True

    def on_init_end(
            self,
            args: RoBaTTrainerArguments,
            state: RoBaTTrainerState,
        ):
        """
        Event called at the end of the initialization of the trainer.
        """
        pass

    def on_train_begin(
            self,
            args: RoBaTTrainerArguments,
            state: RoBaTTrainerState,
        ):
        """
        Event called at the beginning of the training.
        """
        pass
    def on_train_end(
            self,
            args: RoBaTTrainerArguments,
            state: RoBaTTrainerState,
        ):
        """
        Event called at the end of the training.
        """
        pass

    def on_step_begin(
            self,
            args: RoBaTTrainerArguments,
            state: RoBaTTrainerState,
        ):
        """
        Event called at the beginning of each step.
        """
        pass
    def on_step_end(
            self,
            args: RoBaTTrainerArguments,
            state: RoBaTTrainerState,
        ):
        """
        Event called at the end of each step.
        """
        pass

    def on_epoch_begin(
            self,
            args: RoBaTTrainerArguments,
            state: RoBaTTrainerState,
        ):
        """
        Event called at the beginning of each epoch.
        """
        pass
    def on_epoch_end(
            self,
            args: RoBaTTrainerArguments,
            state: RoBaTTrainerState,
        ):
        """
        Event called at the end of each epoch.
        """
        pass

    def on_evaluate(
            self,
            args: RoBaTTrainerArguments,
            state: RoBaTTrainerState,
        ):
        """
        Event called at the end of each evaluation.
        """
        pass

# %%
# allow only one top model to be saved, but can be extended to save top_n models
class SaveTopModelCallback(RoBaTTrainerCallback):
    def __init__(self): 
        super().__init__()
        self.best_models: dict[str, float] = {} # model_path: metric

    def on_init_end(
            self,
            args: RoBaTTrainerArguments,
            state: RoBaTTrainerState,
        ):
        # check assertion
        assert args.metric_for_best_model is not None, "metric_for_best_model should be set"

        self.save_dir = os.path.join(args.output_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.output_dir = args.output_dir
        self.dummy_model_path = os.path.join(self.save_dir, "dummy_model_path")
        if args.greater_is_better:
            self.best_models = {self.dummy_model_path: numpy.finfo(float).min}
        else:
            self.best_models = {self.dummy_model_path: numpy.finfo(float).max}

    def on_step_begin(
            self,
            args: RoBaTTrainerArguments,
            state: RoBaTTrainerState,
        ):
        state.should_save = False

    def on_evaluate(
            self,
            args: RoBaTTrainerArguments,
            state: RoBaTTrainerState,
        ):
        metric = state.metrics[f"eval/{args.metric_for_best_model}"]
        model_path = os.path.join(
            self.save_dir,
            f"ckpt_step-{state.global_step}_{args.metric_for_best_model}-{metric:.4f}",
        )
        operator = numpy.greater if args.greater_is_better else numpy.less
        worst_operator = numpy.argmin if args.greater_is_better else numpy.argmax
        worst_model_idx = worst_operator(list(self.best_models.values()))
        worst_model = list(self.best_models.keys())[worst_model_idx]
        worst_metric = list(self.best_models.values())[worst_model_idx]
        is_model_better = operator(metric, worst_metric)

        self.logger.info(
            f"Evaluation on step {state.global_step}\n"
            f"Current model: {model_path}\n"
            f"Worst saved model: {worst_model}\n"
            f"Current metric: {metric}\n"
            f"Worst saved model metric: {worst_metric}\n"
            f"Is model better: {is_model_better}"
        )

        if not is_model_better:
            return

        if len(self.best_models) >= 1 and is_model_better: # change 1 to top_n is available
            # remove the worst model
            if worst_model == self.dummy_model_path:
                pass
            elif os.path.isdir(worst_model):
                shutil.rmtree(worst_model)
            else:
                os.remove(worst_model)
            del self.best_models[worst_model]

        if len(self.best_models) < 1: # change 1 to top_n is available
            self.best_models[model_path] = metric
            state.should_save = True
            args.output_dir = model_path
        return

    # %%
    @staticmethod
    def get_model_ckpt(model_dirpath: str) -> str:
        """
        Get the model checkpoint from the directory.
        """
        pattern = re.compile(
            r"/ckpt_step-\d+(\.\d+)?_[^/]+-\d+(\.\d+)?$"
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
        return Path(model_ckpt)

class EarlyStoppingCallback(RoBaTTrainerCallback):
    def __init__(self, patience: int=5):
        super().__init__()
        self.patience = patience
        self.best_metric = numpy.finfo(float).max
        self.waiting_steps = 0

    def on_evaluate(
            self,
            args: RoBaTTrainerArguments,
            state: RoBaTTrainerState,
        ):
        metric = state.metrics[f"eval/{args.metric_for_best_model}"]
        operator = numpy.greater if args.greater_is_better else numpy.less
        is_model_better = operator(metric, self.best_metric)

        if is_model_better:
            self.best_metric = metric
            self.waiting_steps = 0
        else:
            self.waiting_steps += 1

        self.logger.info(
            f"Evaluation on step {state.global_step}\n"
            f"Current metric: {metric}\n"
            f"Best metric: {self.best_metric}\n"
            f"Waiting steps: {self.waiting_steps}\n"
            f"Is model better: {is_model_better}"
        )

        if self.waiting_steps >= self.patience:
            state.should_stop = True
        return

# %%
class RoBaTTrainer:
    def __init__(
            self,
            trainer_args: RoBaTTrainerArguments,
            model: PreTrainedModel,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
            train_dataloader: Union[torch.utils.data.DataLoader, Dataset],
            eval_dataloader: Union[torch.utils.data.DataLoader, Dataset],
            compute_metrics: Callable[[RoBaTOutput], Dict[str, float]] = None,
            callbacks: List[RoBaTTrainerCallback] = None,
        ):
        self.trainer_args = trainer_args
        self.accelerator = Accelerator(
            mixed_precision=self.trainer_args.mixed_precision,
            gradient_accumulation_steps=self.trainer_args.gradient_accumulation_steps,
            log_with=self.trainer_args.report_to,
            project_dir=self.trainer_args.output_dir
        )
        if self.accelerator.is_main_process and \
            not os.path.exists(self.trainer_args.output_dir):
            os.makedirs(self.trainer_args.output_dir)

        if self.accelerator.is_main_process:
            config = {
                **asdict(self.trainer_args),
                **model.config.to_dict(),
            }
            self.accelerator.init_trackers(
                "RoBaTTrainer",
                config=config,
            )

        model_kwargs = list(inspect.signature(model.forward).parameters.keys())
        if isinstance(train_dataloader, Dataset):
            select_columns = set(model_kwargs) & set(train_dataloader.column_names)
            train_dataloader = train_dataloader.select_columns(list(select_columns))
            train_dataloader = torch.utils.data.DataLoader(
                train_dataloader,
                batch_size=self.trainer_args.per_device_train_batch_size,
                shuffle=True,
            )
        if isinstance(eval_dataloader, Dataset):
            select_columns = set(model_kwargs) & set(eval_dataloader.column_names)
            eval_dataloader = eval_dataloader.select_columns(list(select_columns))
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataloader,
                batch_size=self.trainer_args.per_device_eval_batch_size,
                shuffle=False,
            )

        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(
            model,
            optimizer,
            lr_scheduler,
            train_dataloader,
            eval_dataloader,
        )

        self.compute_metrics = compute_metrics
        self.callbacks = callbacks if callbacks is not None else []
        self.state = RoBaTTrainerState(
            global_step=0,
            epoch=0.0,
            loss=None,
            learning_rate=None,
            metrics=None,
            should_stop=False,
            should_save=False,
        )
        self.every_state: List[RoBaTTrainerState] = []
        self.every_eval_state: List[RoBaTTrainerState] = []

        self.total_epoch = self.trainer_args.num_train_steps // len(self.train_dataloader) + 1
        self.total_steps = self.trainer_args.num_train_steps

        self.logger = getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.propagate = True # to allow logging to propagate to root logger

        self._run_callbacks("on_init_end", self.trainer_args, self.state)

    def train(self):
        # <--- Training Loop start --->
        self._run_callbacks("on_train_begin", self.trainer_args, self.state)

        progress_bar = tqdm(
            total=self.total_steps
        )
        for _ in range(self.total_epoch):
            # <--- Epoch start --->
            self._run_callbacks("on_epoch_begin", self.trainer_args, self.state)
            for batch in self.train_dataloader:
                # <--- processing bar setup start --->
                _global_step = self.state.copy().global_step # to prevent double counting, copy global_step
                processing_epoch = (self.state.global_step + 1) / len(self.train_dataloader)
                self.state.epoch = processing_epoch
                progress_bar.set_description(f"Training epoch {self.state.epoch:.2f}")
                # <--- processing bar setup end --->
                # <--- Step start --->
                self._run_callbacks("on_step_begin", self.trainer_args, self.state)
                self.state = self.training_step(batch, self.state)
                self.state.global_step = _global_step + 1 # to prevent double counting, add to copied global_step
                progress_bar_postfix = {
                    "loss": self.state.loss,
                    "learning_rate": f"{self.state.learning_rate:.2e}",
                }
                progress_bar.set_postfix(progress_bar_postfix)
                self.every_state.append(self.state)

                if self.state.global_step % self.trainer_args.eval_steps == 0:
                    # <--- Evaluation start --->
                    eval_state = self.evaluate()
                    self.every_eval_state.append(eval_state)
                    # <--- Evaluation end --->
                else:
                    eval_state = None

                self._run_callbacks("on_step_end", self.trainer_args, self.state)
                progress_bar.update(1)
                # <--- Step end --->

                self.state.should_save = (
                    self.state.should_save or
                    (eval_state is not None and eval_state.should_save)
                )
                if self.state.should_save:
                    # <--- Save model start --->
                    self.save_model(self.trainer_args.output_dir)
                    # <--- Save model end --->

                self.state.should_stop = (
                    self.state.should_stop or
                    (self.state.global_step >= self.trainer_args.num_train_steps) or
                    (eval_state is not None and eval_state.should_stop)
                )
                if self.state.should_stop:
                    break
            if self.state.should_stop:
                break
            self._run_callbacks("on_epoch_end", self.trainer_args, self.state)
            # <--- Epoch end --->
        progress_bar.close()

        if self.state.global_step % self.trainer_args.eval_steps != 0:
            # <--- Evaluation start --->
            eval_state = self.evaluate()
            self.every_eval_state.append(eval_state)
            # <--- Evaluation end --->

        self._run_callbacks("on_train_end", self.trainer_args, self.state)
        # <--- Training Loop end --->

        if self.accelerator.is_main_process:
            self.accelerator.end_training()

        for callback in self.callbacks:
            callback.on_train_end(self.trainer_args, self.state)

        self.accelerator.end_training()

    def evaluate(
            self,
            eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
            eval_state: Optional[RoBaTTrainerState] = None
        ):
        eval_dataloader = eval_dataloader if eval_dataloader is not None else self.eval_dataloader
        eval_state = eval_state if eval_state is not None else self.state.copy()
        eval_state = self.evaluation_step(
            eval_dataloader=eval_dataloader,
            eval_state=eval_state,
        )
        self._run_callbacks("on_evaluate", self.trainer_args, eval_state)
        return eval_state

    def training_step(
            self,
            batch: dict[str, torch.Tensor],
            train_state: RoBaTTrainerState
        ) -> RoBaTTrainerState:
        """
        Perform a single training step.
        """
        train_state = train_state.copy()

        with self.accelerator.accumulate(self.model):
            robat_output = self.model(**batch)
            loss = robat_output.loss
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                parameters_to_clip = self.model.parameters()
                self.accelerator.clip_grad_norm_(
                    parameters=parameters_to_clip,
                    max_norm=1.0,
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        train_state.loss = robat_output.loss.detach().item()
        train_state.learning_rate = self.lr_scheduler.get_last_lr()[0]
        train_state.global_step += 1
        self.accelerator.log({
            "train/loss": train_state.loss,
            "train/learning_rate": train_state.learning_rate,
        }, step=train_state.global_step)
        return train_state

    def evaluation_step(
            self,
            eval_dataloader: torch.utils.data.DataLoader,
            eval_state: RoBaTTrainerState
        ) -> RoBaTTrainerState:
        """
        Perform a single evaluation step.
        """
        if self.compute_metrics is None:
            raise ValueError(
                "compute_metrics must be provided to the trainer for evaluation."
            )
        every_outputs = defaultdict(list)
        eval_state = eval_state.copy()
        eval_state.loss = 0.0

        start_time = time.time()
        progress_bar = tqdm(
            total=len(eval_dataloader),
            desc=f"Evaluating on step {eval_state.global_step}",
            leave=True
        )
        with torch.no_grad():
            self.model.eval()
            for batch in eval_dataloader:
                robat_output = self.model(**batch)
                for key, value in robat_output.items():
                    if value is None:
                        continue
                    elif isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    every_outputs[key].append(value)
                progress_bar.update(1)
        every_outputs = {k: numpy.vstack(v) for k, v in every_outputs.items()}
        every_outputs["loss"] = every_outputs["loss"].mean()
        metrics = self.compute_metrics(RoBaTOutput(**every_outputs))
        eval_state.metrics = {f"eval/{k}": v for k, v in metrics.items()}
        eval_state.metrics["eval/loss"] = every_outputs["loss"]
        eval_state.loss = every_outputs["loss"]

        if self.accelerator.is_main_process:
            self.accelerator.log(eval_state.metrics, step=eval_state.global_step)

        progress_bar_postfix = dict(
            (key.replace("eval/", ""), value)
            for key, value in eval_state.metrics.items()
        )
        logger_metric_str = ", ".join([
            f"{key}={progress_bar_postfix[key]:.4f}"
            for key in progress_bar_postfix
        ])
        duration = time.time() - start_time
        progress_bar.set_postfix(**progress_bar_postfix)
        progress_bar.close()

        self.logger.info(
            f"Evaluation on step {self.state.global_step}\n"
            f"with metrics: {logger_metric_str}\n"
            f"with duration: {timedelta(seconds=duration)}"
        )
        return eval_state

    def save_model(self, model_path: str):
        """
        Save the model to the specified path.
        """
        self.model.save_pretrained(model_path)

    def summary(self) -> str:
        """
        Return a summary of the training state.
        """
        for batch in self.train_dataloader:
            break
        model_summary = summary(
            self.model,
            input_data=batch,
            verbose=0
        )
        return str(model_summary)

    def save_state(self, state_dir: str):
        with open(os.path.join(state_dir, "trainer_state.json"), "w") as f:
            json.dump(
                [s.__dict__ for s in self.every_state],
                f,
                indent=4,
                cls=JsonNumpyEncoder,
            )
        with open(os.path.join(state_dir, "trainer_eval_state.json"), "w") as f:
            json.dump(
                [s.__dict__ for s in self.every_eval_state],
                f,
                indent=4,
                cls=JsonNumpyEncoder,
            )

    def _run_callbacks(
            self,
            event: str,
            trainer_args: RoBaTTrainerArguments,
            state: RoBaTTrainerState,
        ):
        for callback in self.callbacks:
            getattr(callback, event)(trainer_args, state)
        return