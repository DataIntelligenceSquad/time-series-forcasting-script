import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
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

args = parser.parse_args()
scaler = Scaler()

def load_data(data_path):
    # Read a pandas DataFrame
    df = pd.read_csv(data_path, delimiter=",")

    if 'SYMBOL' in df.columns:
        df = df.drop('SYMBOL', axis = 1)

    # Create a TimeSeries, specifying the time and value columns
    series = TimeSeries.from_dataframe(df, time_col = "date", fill_missing_dates=True, freq='H')
    series = fill_missing_values(series, fill='auto')

    # Set aside the last 36 months as a validation series
    series, data = series[-args.series_visualize:], series[-args.series_visualize:-args.test_predict]
    data = scaler.fit_transform(data)
    series = scaler.transform(series)
    return series, data

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
    model = RegressionModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
elif model_name == "LinearRegressionModel":
    model = LinearRegressionModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
elif model_name == "LightGBMModel":
    model = LightGBMModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
elif model_name == "CatBoostModel":
    model = CatBoostModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
elif model_name == "XGBModel":
    model = XGBModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
elif model_name == "RNNModel":
    model = RNNModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
elif model_name == "BlockRNNModel":
    model = BlockRNNModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
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
    model = NLinearModel()
else:
    raise ValueError("Invalid model name. Supported models: FFT, TCN")
model = model.load(args.model_path)
# Load the data
series, data = load_data(args.data_path)

# # Normalize the data (using the same scaler as during training)
# scaler = MinMaxScaler()
# data['value'] = scaler.fit_transform(data['value'].values.reshape(-1, 1))

# Perform predictions
predictions = model.predict(args.test_predict + args.future_predict)


series = scaler.inverse_transform(series)
predictions = scaler.inverse_transform(predictions)

# # Denormalize the predictions
# predictions = scaler.inverse_transform(predictions.pd_dataframe()['value'].values.reshape(-1, 1))

series[args.target].plot()
predictions[args.target].plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
plt.legend()


# Save the plot as an image
plt.savefig('predictions_plot.png')
plt.show()


