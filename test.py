import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
from darts.metrics import mape, mse, mae
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
import os
from darts.models import (
    FFT,
    TCNModel,
    VARIMA,
    KalmanForecaster,
    RegressionModel,
    LinearRegressionModel,
    LightGBMModel,
    CatBoostModel,
    XGBModel,
    RNNModel,
    BlockRNNModel,
    NBEATSModel,
    NHiTSModel,
    TransformerModel,
    TFTModel,
    DLinearModel,
    NLinearModel,
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Testing script for predictions')
parser.add_argument("--model_name", type=str, help="Name of the forecasting model to use.")
parser.add_argument('--model_path', type=str, help='Path to the trained model')
parser.add_argument('--data_path', type=str, help='Path to the data file')
parser.add_argument("--input_chunk_size", type=int, default=24, help="Size of the input chunk for TCNModel. Default is 24.")
parser.add_argument("--output_chunk_size", type=int, default=12, help="Size of the output chunk for TCNModel. Default is 12.")
parser.add_argument("--target", type=str, help="Target to visualise")
parser.add_argument("--test_predict", type=int, default=48, help="Size of the predict to test")
parser.add_argument("--future_predict", type=int, default=48, help="Size of the predict to future")
parser.add_argument("--series_visualize", type=int, default=200, help="To visualize the whole test series")
parser.add_argument("--future_only", type=bool, default=False, help="Preidict future.")

args = parser.parse_args()
scaler = Scaler()

# def load_data(data_path):
#     # Read a pandas DataFrame
#     df = pd.read_csv(data_path, delimiter=",")

#     if 'SYMBOL' in df.columns:
#         df = df.drop('SYMBOL', axis = 1)

#     # Create a TimeSeries, specifying the time and value columns
#     series = TimeSeries.from_dataframe(df, time_col = "date", fill_missing_dates=True, freq='H')
#     series = fill_missing_values(series, fill='auto')

#     # Set aside the last 36 months as a validation series
#     series, data = series[-args.series_visualize:], series[-args.series_visualize:-args.test_predict]
#     data = scaler.fit_transform(data)
#     series = scaler.transform(series)
#     return series, data

def load_data(data_path):
    # Read a pandas DataFrame
    df = pd.read_csv(data_path, delimiter=",")
    if 'SYMBOL' in df.columns:
        df = df.drop('SYMBOL', axis = 1)

    # Create a TimeSeries, specifying the time and value columns
    series = TimeSeries.from_dataframe(df, time_col = "date", fill_missing_dates=True, freq='H')
    series = fill_missing_values(series, fill='auto')
    # val = val.fillna('ffill')

    # Set aside the last 36 months as a validation series
    train, val = series[:-args.series_visualize], series[-args.series_visualize:]
    return train, val, series

input_chunk_size = args.input_chunk_size
output_chunk_size = args.output_chunk_size
# Load the trained model
model_name = args.model_name
if model_name == "FFT":
        model = FFT()
elif model_name == "TCN":
    model = TCNModel(input_chunk_length=args.input_chunk_size, output_chunk_length=args.output_chunk_size)
elif model_name == "VARIMA":
        model = VARIMA()
elif model_name == "KalmanForecaster":
    model = KalmanForecaster()
elif model_name == "RegressionModel":
    model = RegressionModel(lags=48, output_chunk_length = args.output_chunk_size)
elif model_name == "LinearRegressionModel":
    model = LinearRegressionModel(lags=48, output_chunk_length = args.output_chunk_size)
elif model_name == "LightGBMModel":
    model = LightGBMModel(lags=48, output_chunk_length = args.output_chunk_size)
elif model_name == "CatBoostModel":
    model = CatBoostModel(lags=48, output_chunk_length = args.output_chunk_size)
elif model_name == "XGBModel":
    model = XGBModel(lags=48, output_chunk_length = args.output_chunk_size)
elif model_name == "RNNModel_rnn":
    model = RNNModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size, model = 'RNN')
elif model_name == "RNNModel_lstm":
    model = RNNModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size, model = 'LSTM')
elif model_name == "RNNModel_gru":
    model = RNNModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size, model = 'GRU')
elif model_name == "BlockRNNModel_rnn":
    model = BlockRNNModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size, model = 'RNN')
elif model_name == "BlockRNNModel_lstm":
    model = BlockRNNModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size, model = 'LSTM')
elif model_name == "BlockRNNModel_gru":
    model = BlockRNNModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size, model = 'GRU')
elif model_name == "NBEATSModel":
    model = NBEATSModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
elif model_name == "NHiTSModel":
    model = NHiTSModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
elif model_name == "TransformerModel":
    model = model = TransformerModel(
            input_chunk_length=args.input_chunk_size,
            output_chunk_length=args.output_chunk_size,
        )
elif model_name == "TFTModel":
    model = TFTModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
elif model_name == "DLinearModel":
    model = DLinearModel()
elif model_name == "NLinearModel":
    model = NLinearModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
else:
    raise ValueError("Invalid model name. Supported models: FFT, TCN")
model = model.load(args.model_path)
# Load the data
train, val, series = load_data(args.data_path)
train = scaler.fit_transform(train)
val = scaler.transform(val)

# Perform predictions
# predictions = model.predict(args.test_predict + args.future_predict)
# print(args.future_only)
if args.future_only:
    predictions = model.predict(series = val, n = args.future_predict)
else:
    predictions = model.predict(n = args.test_predict)


# series = scaler.inverse_transform(train)[-args.series_visualize:]
predictions = scaler.inverse_transform(predictions)
# Tính toán MAPE
mape = mape(series[-len(predictions):][args.target], predictions[args.target])
mse = mse(series[-len(predictions):][args.target], predictions[args.target])
mae = mae(series[-len(predictions):][args.target], predictions[args.target])

# In kết quả
print(f'MAPE: {mape:.4f}')
print(f'MSE: {mse:.4f}')
print(f'MAE: {mae:.4f}')

# # Denormalize the predictions
# predictions = scaler.inverse_transform(predictions.pd_dataframe()['value'].values.reshape(-1, 1))

series[args.target][-args.series_visualize:].plot()
predictions[args.target].plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
plt.legend()


# Save the plot as an image
if not os.path.exists("test_result"):
    # Tạo thư mục "test_result"
    os.makedirs("test_result")
plt.savefig('test_result/predictions_' + args.model_name + '_mape'+ str(mape)+ '_mse'+ str(mse)+ '_mae'+ str(mae)+ '.png')
plt.show()


