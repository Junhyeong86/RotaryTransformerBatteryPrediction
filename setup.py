# %%
# import libraries
from glob import glob
import os
from os.path import basename, splitext
from setuptools import find_packages, setup

# %%
setup(
    name="rotary_transformer_battery_prediction",
    version="0.1",
    packages=["rotary_transformer_battery_prediction"],
    author="Junhyeong",
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "accelerate",
        "datasets",
        "torchinfo",
        "torchdiffeq",
        "tensorboardX"
    ],
    author_email="jun686868@hanyang.ac.kr"
)
