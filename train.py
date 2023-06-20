import argparse
import pandas as pd
from darts import TimeSeries
import os
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

scaler = Scaler()
add_encoders={
            'cyclic': {'future': ['month']},
            'datetime_attribute': {'future': ['hour', 'dayofweek']},
            'position': {'past': ['relative'], 'future': ['relative']},
            'custom': {'past': [lambda idx: (idx.year - 2019) / 50]},
            'transformer': Scaler()
        }
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
    train, val = series[:-args.val_predict], series[-args.val_predict:]
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    series = scaler.transform(series)
    return train, val

def train_model(model_name, data_path, output_path, input_chunk_size, output_chunk_size, num_epochs, verbose):
    # Load the dataset
    train, val = load_data(data_path)
    # print("Train: ", train)
    # print("Val: ", val)

    # Create the model based on the model_name argument
    if model_name == "FFT":
        model = FFT()
        # Fit the model
        model.fit(train)
    elif model_name == "TCN":
        model = TCNModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
        # Fit the model
        model.fit(train, epochs = num_epochs, verbose = verbose)
    elif model_name == "VARIMA":
        model = VARIMA(add_encoders = add_encoders)
        # Fit the model
        model.fit(train)
    elif model_name == "KalmanForecaster":
        model = KalmanForecaster()
        # Fit the model
        model.fit(train)
    elif model_name == "RegressionModel":
        model = RegressionModel(lags=48, output_chunk_length = args.output_chunk_size)
        # Fit the model
        model.fit(train)
    elif model_name == "LinearRegressionModel":
        model = LinearRegressionModel(lags=48, output_chunk_length = args.output_chunk_size)
        # Fit the model
        model.fit(train)
    elif model_name == "LightGBMModel":
        model = LightGBMModel(lags=48, output_chunk_length = args.output_chunk_size)
        # Fit the model
        model.fit(train)
    elif model_name == "CatBoostModel":
        model = CatBoostModel(lags=48, output_chunk_length = args.output_chunk_sỉze)
        # Fit the model
        model.fit(train)
    elif model_name == "XGBModel":
        model = XGBModel(lags=48, output_chunk_length = args.output_chunk_sỉze)
        # Fit the model
        model.fit(train)
    elif model_name == "RNNModel":
        model = RNNModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
        # Fit the model
        model.fit(train, epochs=num_epochs, verbose=verbose)
    elif model_name == "BlockRNNModel":
        model = BlockRNNModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
        # Fit the model
        model.fit(train, epochs=num_epochs, verbose=verbose)
    elif model_name == "NBEATSModel":
        model = NBEATSModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
        # Fit the model
        model.fit(train, epochs=num_epochs, verbose=verbose)
    elif model_name == "NHiTSModel":
        model = NHiTSModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
        # Fit the model
        model.fit(train, epochs=num_epochs, verbose=verbose)
    elif model_name == "TransformerModel":
        model = TransformerModel(
            input_chunk_length=args.input_chunk_size,
            output_chunk_length=args.output_chunk_size,
        )
        # Fit the model
        model.fit(train, epochs=num_epochs, verbose=verbose)
    elif model_name == "TFTModel":
        # add_encoders={
        #     'cyclic': {'future': ['month']},
        #     'datetime_attribute': {'future': ['hour', 'dayofweek']},
        #     'position': {'past': ['relative'], 'future': ['relative']},
        #     'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
        #     'transformer': Scaler()
        # }
        model = TFTModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size, add_encoders = add_encoders)
        # Fit the model
        model.fit(train, epochs=num_epochs, verbose=verbose)
    elif model_name == "DLinearModel":
        model = DLinearModel(input_chunk_length=input_chunk_size, output_chunk_length=output_chunk_size)
        # Fit the model
        model.fit(train)
    elif model_name == "NLinearModel":
        model = NLinearModel()
        # Fit the model
        model.fit(train)
    else:
        raise ValueError("Invalid model name. Supported models: FFT, TCN")

    if not os.path.exists("checkpoints"):
        # Tạo thư mục "checkpoints"
        os.makedirs("checkpoints")
    # Save the trained model
    model.save(output_path)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a forecasting model on a time series dataset.")
    parser.add_argument("model_name", type=str, help="Name of the forecasting model to use.")
    parser.add_argument("data_path", type=str, help="Path to the CSV file containing the time series data.")
    parser.add_argument("output_path", type=str, help="Path to save the trained model.")
    parser.add_argument("--input_chunk_size", type=int, default=24, help="Size of the input chunk for TCNModel. Default is 24.")
    parser.add_argument("--output_chunk_size", type=int, default=12, help="Size of the output chunk for TCNModel. Default is 12.")
    parser.add_argument("--num_epochs", type=int, default=400, help="Number of training epochs. Default is 400.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose training output.")
    parser.add_argument("--val_predict", type=int, default=48, help="Size of the val")

    # Parse the arguments
    args = parser.parse_args()

    # Call the train_model function with the parsed arguments
    train_model(args.model_name, args.data_path, args.output_path, args.input_chunk_size, args.output_chunk_size, args.num_epochs, args.verbose)
