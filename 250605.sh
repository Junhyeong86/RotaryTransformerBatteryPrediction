# #!/bin/bash

# # Pre-train a RoBaT model with SOH optimization
# robat-train --multirun\
#     training.training_args.num_train_steps=10000\
#     training.training_args.learning_rate=5e-5,1e-5,5e-6\
#     model.config.loss_column="state_of_health"\
#     model.config.time_transformer_num_layers=4\
#     model.config.cycle_transformer_num_layers=8\
#     repeat=3

# # Further pre-train a RoBaT model with discharge capacity optimization
# robat-train --multirun\
#     training.training_args.num_train_steps=10000\
#     training.training_args.learning_rate=5e-6,1e-6,5e-7\
#     model.config.loss_column="discharge_capacity"\
#     model.config.time_transformer_num_layers=4\
#     model.config.cycle_transformer_num_layers=8\
#     model.from_pretrained.from_pretrained=true\
#     model.from_pretrained.pretrained_model_name_or_path="best"\
#     repeat=1 # since loading from best pretrained model, no need to repeat

# Fine-tune a SOH optimized RoBaT model with trajectory prediction optimization.
robat-train --multirun\
    training.training_args.num_train_steps=10000\
    training.training_args.learning_rate=5e-5,1e-5,5e-6\
    model="trajectory_prediction"\
    model.config.loss_column="state_of_health"\
    model.config.time_transformer_num_layers=4\
    model.config.cycle_transformer_num_layers=8\
    model.config.ode_prediction_len=128\
    model.config.ode_func_num_layers=2,4\
    model.from_pretrained.from_pretrained=true\
    model.from_pretrained.model_search_dir="./log/train/next_cycle_prediction"\
    model.from_pretrained.pretrained_model_name_or_path="best"\
    repeat=3

# Further traing a trajectory prediction optimized RoBaT model with discharge capacity optimization
robat-train --multirun\
    training.training_args.num_train_steps=10000\
    training.training_args.learning_rate=5e-6,1e-6,5e-7\
    model="trajectory_prediction"\
    model.config.loss_column="discharge_capacity"\
    model.config.time_transformer_num_layers=4\
    model.config.cycle_transformer_num_layers=8\
    model.config.ode_prediction_len=128\
    model.config.ode_func_num_layers=2,4\
    model.from_pretrained.from_pretrained=true\
    model.from_pretrained.pretrained_model_name_or_path="best"\
    repeat=1 # since loading from best pretrained model, no need to repeat