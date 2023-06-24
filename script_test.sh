python test.py \
 --model_name "RNNModel_gru" \
 --model_path "checkpoints/RNNModel_gru.pkl" \
 --data_path "dataset/New_Final_nasdaq_gold_btc.csv" \
 --input_chunk_size 336 \
 --output_chunk_size 48 \
 --target LABEL \
 --future_predict 48 \
 --test_predict 48 \
 --series_visualize 200 \
#  --future_only true \