#!/bin/bash

# Set the paths and filenames
data_path="dataset/Final_nasdaq_gold_btc.csv"
output_path="checkpoints/model.pkl"

# Set the model name and other parameters
# Current can not run VARIMA
# TCNModel,VARIMA,KalmanForecaster,RegressionModel,LinearRegressionModel,
# LightGBMModel, CatBoostModel, XGBModel,RNNModel,BlockRNNModel,
# NBEATSModel, NHiTSModel,TransformerModel,TFTModel,DLinearModel,NLinearModel,
model_name="TransformerModel"
input_chunk_size=96
output_chunk_size=48
num_epochs=10
val_predict=48
verbose="--verbose"  # Uncomment this line to enable verbose output

# Run the training script
python train.py "$model_name" "$data_path" "$output_path" \
    --input_chunk_size "$input_chunk_size" \
    --output_chunk_size "$output_chunk_size" \
    --num_epochs "$num_epochs" \
    --val_predict "$val_predict"
    $verbose
