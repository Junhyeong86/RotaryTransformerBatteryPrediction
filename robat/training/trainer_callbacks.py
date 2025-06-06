# %%
import os
import re
import shutil
from pathlib import Path
from typing import Optional
from logging import getLogger
from textwrap import dedent

import numpy

from datetime import datetime

from transformers import (
    TrainerCallback,
)

# %%
class SaveTopModelsCallback(TrainerCallback):
    """
    A callback that saves the top n models based on the evaluation metric.
    """
    def __init__(
            self,
            top_n: int=3,
        ):
        """
        Args:
            top_n: int
                - The number of top models to save.

        Raises:
            TypeError:
                - If Type is not correct.
        """
        # Check the types of the arguments
        self.top_n: int = top_n
        self.logger = getLogger(self.__module__ + "." + self.__class__.__name__)
        # Check the values of the arguments
        if not isinstance(self.top_n, int):
            raise TypeError(f"top_n must be an integer, not {type(self.top_n)}")
        assert self.top_n > 0, "top_n must be greater than 0"

        self.best_models: dict[str, float] = {} # {model_path: metric}

    def on_train_begin(
            self,
            args,
            state,
            control,
            **kwargs
        ):
        # check assertion
        assert args.metric_for_best_model is not None, "metric_for_best_model must be provided"

        self.save_dir = os.path.join(args.output_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.output_dir = args.output_dir
        args.save_strategy = "no"
        args.num_train_epochs += 1
        # one more step should be executed for saving checkpoints
        args.max_steps = state.max_steps + 1

        self.dummy_model_path = os.path.join(self.save_dir, f"ckpt_epoch-0")
        if args.greater_is_better:
            self.best_models = {self.dummy_model_path: numpy.finfo(float).min}
        else:
            self.best_models = {self.dummy_model_path: numpy.finfo(float).max}

    def on_epoch_begin(
            self,
            args,
            state,
            control,
            **kwargs
        ):
        args.output_dir = self.output_dir

    def on_evaluate(
            self,
            args,
            state,
            control,
            **kwargs
        ):
        if getattr(self, "save_dir", None) is None:
            return
        monitor = args.metric_for_best_model
        self.logger.info(kwargs["metrics"])
        # Get the latest evaluation metric
        metric = kwargs["metrics"]["eval_" + monitor] # The evaluation metric

        # Get the current model path
        if state.epoch is None:
            state.epoch = 0
        model_path = os.path.join(self.save_dir, f"ckpt_epoch-{state.epoch:.2f}_{monitor}-{metric:.2f}")
        operator = numpy.greater if args.greater_is_better else numpy.less
        worst_operator = numpy.argmin if args.greater_is_better else numpy.argmax
        worst_model_idx = worst_operator(list(self.best_models.values()))
        worst_model = list(self.best_models.keys())[worst_model_idx]
        worst_metric = list(self.best_models.values())[worst_model_idx]
        is_model_better = operator(metric, worst_metric)

        # logging the evaluation
        self.logger.info(dedent(f"""\
            epoch {state.epoch} evaluation {monitor}
            Current model: {model_path}
            Worst saved model: {worst_model}
            Current metric: {metric}
            Worst saved model metric: {worst_metric}
        """))

        # If the current model is better than the worst model in the top n models
        if len(self.best_models) >= self.top_n and is_model_better:
            # Remove the worst model
            if worst_model == self.dummy_model_path:
                pass
            elif os.path.isdir(worst_model):
                shutil.rmtree(worst_model)
            else:
                os.remove(worst_model)
            del self.best_models[worst_model]

        # If the top n models are not filled
        if len(self.best_models) < self.top_n:
            # Save the current model
            self.best_models[model_path] = metric
            control.should_save = True
            args.output_dir = model_path

    # %%
    @staticmethod
    def get_model_ckpt(model_dirpath: str) -> str:
        """
        Get the model checkpoint from the directory.
        """
        pattern = re.compile(
            r"/ckpt_epoch-\d+(\.\d+)?_[^/]+-\d+(\.\d+)?/checkpoint-\d+$"
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