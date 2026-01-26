# RoBaT: Rotary Transformer for Battery Prediction

**RoBaT** is a deep learning framework that combines Rotary Transformers with Neural ODEs to predict Battery State of Health (SOH), discharge capacity, and future degradation trajectories.

## Installation

### Basic Installation

To use the `robat` package for inference or as a library:

```sh
pip install "robat @ git+https://github.com/Junhyeong86/RotaryTransformerBatteryPrediction.git"
```

### Full Installation (for Training)

To train or fine-tune your own models, install with the `[train]` extra:

```sh
pip install "robat[train] @ git+https://github.com/Junhyeong86/RotaryTransformerBatteryPrediction.git"
```

---

## Training your own RoBaT model

RoBaT utilizes **Hydra** for configuration management. You can override any parameter defined in `robat/config` via CLI arguments.

### 1. SOH Prediction Optimization

In this stage, the model is trained to predict the State of Health (SOH) using a Rotary Transformer backbone.

#### **Pre-train a RoBaT model with SOH optimization**

- We use the polynomial learning rate scheduler to accelerate initial convergence.
- The `repeat=3` argument runs the experiment 3 times to ensure statistical reliability.

```sh
robat-train --multirun \
    training/lr_scheduler="polynominal" \
    training.trainer_args.num_train_steps=10000 \
    training.optimizer.lr=5e-5 \
    training.ds.refs_to_remove="['KIER']" \
    model.config.loss_column="state_of_health" \
    model.config.time_transformer_num_layers=4 \
    model.config.cycle_transformer_num_layers=8 \
    repeat=3
```

#### **Further pre-train with Discharge Capacity optimization**

- We fine-tune the SOH-optimized model using a smaller learning rate to stabilize the discharge capacity prediction.
- `pretrained_model_name_or_path="best"` automatically searches and loads the best checkpoint from the previous step.

```sh
robat-train --multirun \
    training.trainer_args.num_train_steps=10000 \
    training.optimizer.lr=1e-6 \
    training.ds.refs_to_remove="['KIER']" \
    model.config.loss_column="discharge_capacity" \
    model.config.time_transformer_num_layers=4 \
    model.config.cycle_transformer_num_layers=8 \
    model.from_pretrained.from_pretrained=true \
    model.from_pretrained.pretrained_model_name_or_path="best" \
    repeat=3
```

### 2. Trajectory Prediction Optimization

This stage integrates Neural ODE layers to predict continuous degradation paths.

#### **Fine-tune for Trajectory Prediction**

- Since ODE layers are randomly initialized, `repeat=3` is essential to find the best performing weights.
- `model.from_pretrained.model_search_dir` directs the script to find the best SOH model as a starting point.

```sh
robat-train --multirun \
    training/lr_scheduler="polynominal" \
    training.trainer_args.num_train_steps=10000 \
    training.optimizer.lr=5e-5 \
    training.ds.refs_to_remove="['KIER']" \
    model="trajectory_prediction" \
    model.config.loss_column="state_of_health" \
    model.config.time_transformer_num_layers=4 \
    model.config.cycle_transformer_num_layers=8 \
    model.config.ode_prediction_len=128 \
    model.config.ode_func_num_layers=4 \
    model.from_pretrained.from_pretrained=true \
    model.from_pretrained.model_search_dir="./log/next_cycle_prediction" \
    model.from_pretrained.pretrained_model_name_or_path="best" \
    repeat=3
```

#### **Further train with Discharge Capacity optimization**

- Fine-tune survival/degradation trajectories for capacity prediction.

```sh
robat-train --multirun \
    training.trainer_args.num_train_steps=10000 \
    training.optimizer.lr=1e-6 \
    training.ds.refs_to_remove="['KIER']" \
    model="trajectory_prediction" \
    model.config.loss_column="discharge_capacity" \
    model.config.time_transformer_num_layers=4 \
    model.config.cycle_transformer_num_layers=8 \
    model.config.ode_prediction_len=128 \
    model.config.ode_func_num_layers=4 \
    model.from_pretrained.from_pretrained=true \
    model.from_pretrained.pretrained_model_name_or_path="best" \
    repeat=3
```

---

## Evaluate your RoBaT model

After training, you can evaluate the model ensemble (repeats) and generate a summary report:

```sh
# Evaluate all models saved in the log directory
robat-eval ./log
```

The evaluation results will be saved as `evaluation_results.csv`.
