# %%
# register eval resolver as default for OmegaConf
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    "eval",
    eval,
    use_cache=False
)
