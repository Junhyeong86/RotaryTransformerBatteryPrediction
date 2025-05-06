# %%
import random
random.seed(42) # for reproducibility

# %%
from transformers.utils import logging

logging.set_verbosity_error()
logging.enable_progress_bar()
