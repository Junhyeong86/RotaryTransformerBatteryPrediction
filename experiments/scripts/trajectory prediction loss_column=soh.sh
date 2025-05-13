#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"
cd "../../"

# Define arrays for models, datasets, and learning rates
ode_func_num_layers=(2 4)
learning_rate=(5e-5 1e-5)
learning_rate_scheduler=(cosine)
num_training=3

pass_directories=()

# Iterate over model checkpoints
for ode_func_nl in "${ode_func_num_layers[@]}"; do
    for lr in "${learning_rate[@]}"; do
        for lr_scheduler in "${learning_rate_scheduler[@]}"; do
            for ((i=0; i<num_training; i++)); do
                all_directories=()
                # Print training details
                echo "--------------------------------"
                echo "ode_func_num_layers=$ode_func_nl"
                echo "learning_rate=$lr"
                echo "lr_scheduler=$lr_scheduler"
                echo "training=$i"

                # Add your training command here, for example:
                python3 "./experiments/train.py" \
                    --model_purpose "trajectory_prediction" \
                    --time_transformer_num_layers 4 \
                    --cycle_transformer_num_layers 8 \
                    --ode_func_num_layers "$ode_func_nl" \
                    --learning_rate "$lr" \
                    --lr_scheduler "$lr_scheduler" \
                    --loss_column "state_of_health" \
                    --training_dir_suffix "$i"
            done
        done
    done
done