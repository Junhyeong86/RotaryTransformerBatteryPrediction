# %%
# register eval resolver as default for OmegaConf
from omegaconf import OmegaConf

if not OmegaConf.has_resolver("eval"):
    OmegaConf.clear_resolver("eval")
OmegaConf.register_new_resolver(
    "eval",
    eval,
    use_cache=False
)
