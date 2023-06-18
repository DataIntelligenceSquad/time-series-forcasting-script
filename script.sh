#!/bin/bash

# Set the paths and filenames
data_path="dataset/gold_price_hourly.csv"
output_path="checkpoints/model.pkl"

# Set the model name and other parameters
model_name="TCN"
input_chunk_size=24
output_chunk_size=12
num_epochs=10
verbose="--verbose"  # Uncomment this line to enable verbose output

# Run the training script
python train.py "$model_name" "$data_path" "$output_path" \
    --input_chunk_size "$input_chunk_size" \
    --output_chunk_size "$output_chunk_size" \
    --num_epochs "$num_epochs" \
    $verbose
