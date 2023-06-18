python test.py \
 --model_name "TCN" \
 --model_path "checkpoints/model.pkl" \
 --data_path "dataset/gold_price_hourly.csv" \
 --input_chunk_size 24 \
 --output_chunk_size 12 \