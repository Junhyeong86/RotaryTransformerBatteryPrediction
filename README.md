## Install
```sh
pip install "robat @ git+https://github.com/Junhyeong86/RotaryTransformerBatteryPrediction.git"
```

## Training your own RoBaT model
### Install
To train your own RoBaT model, you need to install the `robat` package with `train` extra.
```sh
pip install "robat[train] @ git+https://github.com/Junhyeong86/RotaryTransformerBatteryPrediction.git"
```
### SOH Prediction Optimization
- **Pre-train a RoBaT model with SOH optimization**
- to fasten the training, we used the `training/lr_scheduler="polynominal"`.
```sh
robat-train --multirun\
    training/lr_scheduler="polynominal"\
    training.trainer_args.num_train_steps=10000\
    training.optimizer.lr=5e-5\
    training.ds.ref_filter_list='["MATR"]',null\
    model.config.loss_column="state_of_health"\
    model.config.time_transformer_num_layers=4\
    model.config.cycle_transformer_num_layers=8\
    repeat=3
```
- **Further pre-train a RoBaT model with discharge capacity optimization**
- To stabilize the training, we use a smaller learning rate than the SOH optimization.
```sh
robat-train --multirun\
    training.trainer_args.num_train_steps=10000\
    training.optimizer.lr=1e-6\
    model.config.loss_column="discharge_capacity"\
    model.config.time_transformer_num_layers=4\
    model.config.cycle_transformer_num_layers=8\
    model.from_pretrained.from_pretrained=true\
    model.from_pretrained.pretrained_model_name_or_path="best"
```
### Trajectory Prediction Optimization
- **Fine-tune a SOH optimized RoBaT model with trajectory prediction optimization.**
- The model is searched from the best SOH optimized model, thereby setting `model.from_pretrained.model_search_dir` to the SOH optimization log directory.
- Even if model is fine-tuned with ode_prediction_len=128, it can also be used for longer or shorter prediction lengths. i.e., it is just a hyperparameter for training, not a hard constraint.
- Unlike the SOH further pre-training, this fine-tuning is done with repeat=3 to get the best model. Because there is no ode layers in the SOH optimized model, ode layers are randomly initialized which means repeat is needed to get the best model.
```sh
robat-train --multirun\
    training.trainer_args.num_train_steps=10000\
    training.optimizer.lr=5e-5\
    model="trajectory_prediction"\
    model.config.loss_column="state_of_health"\
    model.config.time_transformer_num_layers=4\
    model.config.cycle_transformer_num_layers=8\
    model.config.ode_prediction_len=128\
    model.config.ode_func_num_layers=4\
    model.from_pretrained.from_pretrained=true\
    model.from_pretrained.model_search_dir="./log/train/next_cycle_prediction"\
    model.from_pretrained.pretrained_model_name_or_path="best"\
    repeat=3
```
- **Further traing a trajectory prediction optimized RoBaT model with discharge capacity optimization**
- To stabilize the training, we use a smaller learning rate than the trajectory prediction optimization.
```sh
robat-train --multirun\
    training.trainer_args.num_train_steps=10000\
    training.optimizer.lr=1e-6\
    model="trajectory_prediction"\
    model.config.loss_column="discharge_capacity"\
    model.config.time_transformer_num_layers=4\
    model.config.cycle_transformer_num_layers=8\
    model.config.ode_prediction_len=128\
    model.config.ode_func_num_layers=4\
    model.from_pretrained.from_pretrained=true\
    model.from_pretrained.pretrained_model_name_or_path="best"
```

## Evaluate your own RoBaT model
### Install
To evaluate your own RoBaT model, you need to install the `robat` package with `train` extra.
```sh
pip install "robat[train] @ git+https://github.com/Junhyeong86/RotaryTransformerBatteryPrediction.git"
```
### Evaluation
```sh
robat-eval ./log
```