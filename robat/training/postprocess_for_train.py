# %%
import os
import re
import json
from ast import literal_eval
from typing import Union, Dict
from collections import defaultdict
from pathlib import Path

import pandas
from omegaconf import OmegaConf, DictConfig

# %%
def get_model_essential_hyperparams(
        cfg: Union[str, Dict, DictConfig],
        include_ode_layers: bool = True,
    ) -> Dict[str, Union[int, str]]:
    """
    Get essential hyperparameters from the model configuration.
    """
    if isinstance(cfg, str):
        cfg = literal_eval(cfg)

    essential_hyperparams = {
        "embedding_dim": cfg["embedding_dim"],
        "feed_forward_hidden_dim": cfg["feed_forward_hidden_dim"],
        "num_heads": cfg["num_heads"],
        "time_transformer_num_layers": cfg["time_transformer_num_layers"],
        "random_time_len": cfg["random_time_len"],
        "ref_time_seq_len": cfg["ref_time_seq_len"],
        "time_transformer_output_dim": cfg["time_transformer_output_dim"],
        "cycle_transformer_num_layers": cfg["cycle_transformer_num_layers"],
        "freq_base": cfg["freq_base"],
        "padding_side": cfg["padding_side"],
    }
    if include_ode_layers:
        essential_hyperparams["ode_func_num_layers"] = cfg["ode_func_num_layers"]
    
    return essential_hyperparams

# %%
def is_compatable_model_config(
        cfg1: Union[str, Dict, DictConfig],
        cfg2: Union[str, Dict, DictConfig],
        ode_layer_check: bool = True,
    ) -> bool:
    if isinstance(cfg1, str):
        cfg1 = literal_eval(cfg1)
    if isinstance(cfg2, str):
        cfg2 = literal_eval(cfg2)

    cfg1_essential = get_model_essential_hyperparams(cfg1, include_ode_layers=ode_layer_check)
    cfg2_essential = get_model_essential_hyperparams(cfg2, include_ode_layers=ode_layer_check)

    for cfg1_ess, cfg2_ess in zip(cfg1_essential.values(), cfg2_essential.values()):
        if cfg1_ess != cfg2_ess:
            return False
    return True

# %%
def get_eval_df(log_dir: str) -> pandas.DataFrame:
    """
    Get the evaluation dataframe.
    """
    eval_df = defaultdict(list)
    log_dir = log_dir if log_dir else "log"
    for root, _, files in os.walk(log_dir):
        root = Path(root)
        if any(str(p).startswith("_") for p in root.parts):
            continue  # skip hidden directories
        for file in files:
            if file != "trainer_eval_state.json":
                continue
            if re.search(r"repeat=(\d+)", str(root.name)) is None:
                continue
            trainer_eval_state = json.load(open(root/file))
            cfg_path = root.resolve().parents[0] / "config.yaml"
            cfg = OmegaConf.load(cfg_path)
            for log in trainer_eval_state:
                eval_df["dirpath"].append(str(root))
                for key, value in log["metrics"].items():
                    eval_df[key].append(value)
                eval_df["model.config"].append(str(cfg.model.config))
                eval_df["model.model_purpose"].append(cfg.model.model_purpose)
    eval_df = pandas.DataFrame(eval_df)
    return eval_df