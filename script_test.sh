python test.py \
 --model_name "TransformerModel" \
 --model_path "checkpoints/model.pkl" \
 --data_path "dataset/Final_nasdaq_gold_btc.csv" \
 --input_chunk_size 24 \
 --output_chunk_size 12 \
 --target LABEL \