#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"
cd "../../"

# Define arrays for models, datasets, and learning rates
time_transformer_num_layers=(4)
cycle_transformer_num_layers=(8)
learning_rate=(5e-5 1e-5 5e-6 1e-6 5e-7 1e-7 5e-8)
learning_rate_scheduler=(cosine linear)
num_training=1

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
                    python3 "./experiments/next cycle prediction/train.py" \
                        "next_cycle_prediction" \
                        --time_transformer_num_layers "$time_transform_nl" \
                        --cycle_transformer_num_layers "$cycle_transform_nl" \
                        --learning_rate "$lr" \
                        --lr_scheduler "$lr_scheduler" \
                        --loss_column "discharge_capacity" \
                        --training_dir_suffix "$i"
                done
            done
        done
    done
done
