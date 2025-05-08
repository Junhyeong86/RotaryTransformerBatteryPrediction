#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"
cd "../../"

# Define arrays for models, datasets, and learning rates
time_transformer_num_layers=(4)
cycle_transformer_num_layers=(4 8)
learning_rate=(5e-5 5e-7)
learning_rate_scheduler=(cosine)
num_training=3

pass_directories=()

# Iterate over model checkpoints
for time_transform_nl in "${time_transformer_num_layers[@]}"; do
    for cycle_transform_nl in "${cycle_transformer_num_layers[@]}"; do
        for lr in "${learning_rate[@]}"; do
            for lr_scheduler in "${learning_rate_scheduler[@]}"; do
                for ((i=0; i<num_training; i++)); do
                    all_directories=()
                    # Print training details
                    echo "--------------------------------"
                    echo "time_transformer_num_layers=$time_transform_nl"
                    echo "cycle_transformer_num_layers=$cycle_transform_nl"
                    echo "learning_rate=$lr"
                    echo "lr_scheduler=$lr_scheduler"
                    echo "training=$i"

                    # Add your training command here, for example:
                    python3 "./experiments/SOH/train.py" \
                        --time_transformer_num_layers "$time_transform_nl" \
                        --cycle_transformer_num_layers "$cycle_transform_nl" \
                        --learning_rate "$lr" \
                        --lr_scheduler "$lr_scheduler" \
                        --training_dir_suffix "$i"
                done
            done
        done
    done
done
