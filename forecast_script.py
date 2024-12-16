from plotting_utils import visualize_data, plot_results, plot_training_history
from utils import create_seasonal_features_filtered, load_data, filter_last_n_years, prepare_seasonal_data

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Dense

def plot_line_graph(x_data_list, y_data_list, labels, title, x_label, y_label, legend_labels, output_path, figure_size=(14,7)):
    plt.figure(figsize=figure_size)
    for x_data, y_data, label in zip(x_data_list, y_data_list, legend_labels):
        plt.plot(x_data, y_data, label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_heatmap(data, title, x_tick_labels, y_tick_labels, output_path, figure_size=(12,10)):
    plt.figure(figsize=figure_size)
    plt.matshow(data, fignum=1, cmap='coolwarm')
    plt.xticks(range(len(x_tick_labels)), x_tick_labels, rotation=90)
    plt.yticks(range(len(y_tick_labels)), y_tick_labels)
    plt.colorbar()
    plt.title(title, pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def visualize_data(data, currency, output_dir):
    close_col = f'Close_{currency}'
    plot_path = os.path.join(output_dir, f'{currency}_closing_prices.png')
    plot_line_graph(
        x_data_list=[data.index],
        y_data_list=[data[close_col]],
        labels=[close_col],
        title=f'{currency} Closing Prices',
        x_label='Date',
        y_label='Price',
        legend_labels=[close_col],
        output_path=plot_path,
        figure_size=(14, 7)
    )
    numeric_data = data.select_dtypes(include=[np.number])
    correlation = numeric_data.corr()
    heatmap_path = os.path.join(output_dir, f'{currency}_correlation_heatmap.png')
    plot_heatmap(
        data=correlation,
        title='Correlation Heatmap',
        x_tick_labels=correlation.columns,
        y_tick_labels=correlation.columns,
        output_path=heatmap_path,
        figure_size=(12, 10)
    )

def decompose_time_series(data, currency, output_dir):
    close_col = f'Close_{currency}'
    result = seasonal_decompose(data[close_col], model='multiplicative', period=252)
    plt.figure(figsize=(15, 10))
    plt.subplot(411)
    plt.title('Original Time Series')
    plt.plot(data.index, result.observed)
    plt.subplot(412)
    plt.title('Trend')
    plt.plot(data.index, result.trend)
    plt.subplot(413)
    plt.title('Seasonal')
    plt.plot(data.index, result.seasonal)
    plt.subplot(414)
    plt.title('Residual')
    plt.plot(data.index, result.resid)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{currency}_seasonal_decomposition.png'))
    plt.close()
    return result

def plot_results(currency, y_train_seq, y_test_plot, predictions, y_train_dates, y_test_dates, scaler_y, currency_output_dir, dataset_time):
    y_train_plot = scaler_y.inverse_transform(y_train_seq.reshape(-1, 1)).flatten()
    forecast_plot_path = os.path.join(currency_output_dir, f'{currency}_forecast.png')
    plot_line_graph(
        x_data_list=[y_train_dates, y_test_dates, y_test_dates],
        y_data_list=[y_train_plot, y_test_plot, predictions],
        labels=['Training Data', 'Actual Prices', 'Predicted Prices'],
        title=f'{currency} Closing Price Prediction (Last {dataset_time} Years)',
        x_label='Date',
        y_label='Price',
        legend_labels=['Training Data', 'Actual Prices', 'Predicted Prices'],
        output_path=forecast_plot_path,
        figure_size=(20, 10)
    )
    comparison_plot_path = os.path.join(currency_output_dir, f'{currency}_actual_vs_predicted.png')
    plot_line_graph(
        x_data_list=[y_test_dates, y_test_dates],
        y_data_list=[y_test_plot, predictions],
        labels=['Actual Prices', 'Predicted Prices'],
        title=f'{currency} Actual vs Predicted Prices (Test Set)',
        x_label='Date',
        y_label='Price',
        legend_labels=['Actual Prices', 'Predicted Prices'],
        output_path=comparison_plot_path,
        figure_size=(20, 10)
    )

def plot_training_history(history, output_dir, currency):
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    if not loss:
        return
    plt.figure(figsize=(14, 7))
    plt.plot(loss, label='Training Loss')
    if val_loss:
        plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{currency} Training History')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{currency}_training_history.png')
    plt.savefig(plot_path)
    plt.close()

class DetailedLoggingCallback(Callback):
    def __init__(self, model_counter, total_models):
        super().__init__()
        self.model_counter = model_counter
        self.total_models = total_models

    def on_train_begin(self, logs=None):
        print(f"\nStarting training model {self.model_counter.count} of {self.total_models}")

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            val_loss = logs.get('val_loss')
            print(f"Model {self.model_counter.count} | Epoch {epoch + 1}: val_loss: {val_loss:.4f}")

class ModelCounter:
    def __init__(self, total):
        self.count = 0
        self.total = total

    def increment(self):
        self.count += 1
        return self.count

def create_rnn_model(rnn_type='LSTM', n_layers=1, units=50, activation='tanh', optimizer='adam', input_shape=None):
    if input_shape is None:
        raise ValueError("input_shape must be specified")
    model = Sequential()
    for i in range(n_layers):
        return_sequences = True if i < n_layers - 1 else False
        if rnn_type == 'LSTM':
            model.add(
                LSTM(units=units, activation=activation, input_shape=input_shape, return_sequences=return_sequences))
        elif rnn_type == 'GRU':
            model.add(
                GRU(units=units, activation=activation, input_shape=input_shape, return_sequences=return_sequences))
        else:
            model.add(SimpleRNN(units=units, activation=activation, input_shape=input_shape, return_sequences=return_sequences))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def create_lstm_sequences(X, y, sequence_length=1):
    X_seq = []
    y_seq = []
    for i in range(len(X) - sequence_length):
        X_seq.append(X.iloc[i:i + sequence_length].values)
        y_seq.append(y.iloc[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)

def load_and_prepare_data(currency, combined_data_dir, output_dir, scaling_method, dataset_time, short_term_lag, long_term_lag):
    file_name = f'combined_data_{currency}.csv'
    file_path = os.path.join(combined_data_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Combined data file for {currency} not found at {file_path}.")
        return None, None, None, None
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    if data is None or data.empty:
        print(f"No data available for {currency}.")
        return None, None, None, None
    data = data.drop(columns=['Day_of_Week'], errors='ignore')
    data = prepare_seasonal_data(data, currency, output_dir)
    filtered_data = filter_last_n_years(data, n_years=dataset_time)
    if filtered_data.empty:
        print(f"Filtered data for {currency} is empty.")
        return None, None, None, None
    scaler_options = {
        'standard': StandardScaler(),
        'normalize': MinMaxScaler(),
        'robust': RobustScaler()
    }
    scaler_X = scaler_options.get(scaling_method, StandardScaler())
    scaler_y = StandardScaler()
    X, y = create_seasonal_features_filtered(
        filtered_data,
        currency=currency,
        use_short_term_lag=True,
        use_long_term_lag=True,
        short_term_lag=short_term_lag,
        long_term_lag=long_term_lag
    )
    if X.empty or y.empty:
        print(f"No features/target available for {currency}.")
        return None, None, None, None
    return X, y, scaler_X, scaler_y

def train_rnn_models(X_train_seq, y_train_seq, X_val_seq, y_val_seq, sequence_length, param_grid, model_counter, total_models):
    results = []
    histories = []
    from itertools import product
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    param_combinations = list(product(*values))
    for params in param_combinations:
        param_dict = dict(zip(keys, params))
        model_counter.increment()
        print(f"\nTraining model {model_counter.count} of {total_models}: {param_dict}")
        model = create_rnn_model(
            rnn_type=param_dict['rnn_type'],
            n_layers=param_dict['n_layers'],
            units=param_dict['units'],
            activation=param_dict['activation'],
            optimizer=param_dict['optimizer'],
            input_shape=(sequence_length, X_train_seq.shape[2])
        )
        logging_callback = DetailedLoggingCallback(model_counter, total_models)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=param_dict['epochs'],
            batch_size=param_dict['batch_size'],
            callbacks=[early_stopping],
            verbose=1
        )
        val_mse = min(history.history['val_loss'])
        results.append({
            'params': param_dict,
            'val_mse': val_mse,
            'history': history
        })
        histories.append(history)
    return results, histories

def plot_validation_losses(histories, param_combinations, currency, output_dir):
    plt.figure(figsize=(20, 10))
    for history, params in zip(histories, param_combinations):
        epochs = range(1, len(history.history['val_loss']) + 1)
        plt.plot(epochs, history.history['val_loss'], label=str(params))
    plt.title(f'Validation Loss per Epoch for Different Model Configurations - {currency}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{currency}_validation_losses_per_epoch.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Validation losses per epoch plot saved to {plot_path}")

def make_predictions(best_model, X_test_seq, y_test_seq, scaler_y):
    predictions_scaled = best_model.predict(X_test_seq)
    predictions_scaled = pd.Series(predictions_scaled.flatten()).rolling(window=5, min_periods=1).mean().values
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    y_test_plot = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    return predictions, y_test_plot

def forecast_currency_with_seasonal_data(sequence_length, currency, combined_data_dir, output_dir,
                                         scaling_method='standard', forecasting_type='recursive',
                                         dataset_time=6, prediction_time=180,
                                         short_term_lag=60, long_term_lag=360, param_grid=None):
    currency_output_dir = os.path.join(output_dir, currency)
    os.makedirs(currency_output_dir, exist_ok=True)
    X, y, scaler_X, scaler_y = load_and_prepare_data(currency, combined_data_dir, output_dir, scaling_method,
                                                     dataset_time, short_term_lag, long_term_lag)
    if X is None:
        return
    visualize_data(pd.concat([X, y.rename('Close_' + currency)], axis=1), currency, currency_output_dir)
    if forecasting_type == 'recursive':
        test_period_days = prediction_time
        train_size = len(X) - test_period_days - sequence_length
        if train_size <= 0:
            print(f"Not enough data to train for {currency}.")
            return
        X_train_df = X.iloc[:train_size + sequence_length]
        X_val_df = X.iloc[train_size:]
        y_train_series = y.iloc[:train_size + sequence_length]
        y_val_series = y.iloc[train_size:]
        scaler_X.fit(X_train_df)
        X_train_scaled = scaler_X.transform(X_train_df)
        X_val_scaled = scaler_X.transform(X_val_df)
        scaler_y.fit(y_train_series.values.reshape(-1, 1))
        y_train_scaled = scaler_y.transform(y_train_series.values.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val_series.values.reshape(-1, 1)).flatten()
        X_train_seq, y_train_seq = create_lstm_sequences(
            pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train_df.index),
            pd.Series(y_train_scaled, index=y_train_series.index),
            sequence_length
        )
        X_val_seq, y_val_seq = create_lstm_sequences(
            pd.DataFrame(X_val_scaled, columns=X.columns, index=X_val_df.index),
            pd.Series(y_val_scaled, index=y_val_series.index),
            sequence_length
        )
        if len(X_val_seq) == 0:
            print(f"Not enough validation sequences after adjustment for {currency}.")
            return
        total_models = 1
        for v in param_grid.values():
            total_models *= len(v)
        model_counter = ModelCounter(total_models)
        print(f"Starting manual grid search for {currency} (Recursive Forecasting with RNN)...")
        results, histories = train_rnn_models(
            X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            sequence_length, param_grid, model_counter, total_models
        )
        param_combinations = [result['params'] for result in results]
        plot_validation_losses([result['history'] for result in results], param_combinations, currency, currency_output_dir)
        print(f"Completed manual grid search for {currency}.")
        best_result = min(results, key=lambda x: x['val_mse'])
        best_params = best_result['params']
        best_history = best_result['history']
        print(f"Best Parameters for {currency} RNN: {best_params}")
        print(f"Best Validation MSE for {currency} RNN: {best_result['val_mse']}")
        best_model = create_rnn_model(
            rnn_type=best_params['rnn_type'],
            n_layers=best_params['n_layers'],
            units=best_params['units'],
            activation=best_params['activation'],
            optimizer=best_params['optimizer'],
            input_shape=(sequence_length, X_train_seq.shape[2])
        )
        best_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
            ],
            verbose=0
        )
        predictions, y_val_plot = make_predictions(best_model, X_val_seq, y_val_seq, scaler_y)
        y_test_dates = y_val_series.index[sequence_length:]
        assert len(y_val_seq) == len(predictions) == len(y_test_dates), "Lengths of test data do not match."
        plot_results(
            currency, y_train_seq, y_val_plot, predictions,
            y_train_series.index[sequence_length:], y_test_dates,
            scaler_y, currency_output_dir, dataset_time
        )
        mse = mean_squared_error(y_val_plot, predictions)
        rmse = np.sqrt(mse)
        print(f'Mean Squared Error for {currency}: {mse}')
        print(f'Root Mean Squared Error for {currency}: {rmse}')
        mse_file = os.path.join(currency_output_dir, f'{currency}_mse.txt')
        with open(mse_file, 'w') as f:
            header = "Rank, Parameters, Mean MSE, RMSE\n"
            f.write(header)
            sorted_results = sorted(results, key=lambda x: x['val_mse'])
            for rank, result in enumerate(sorted_results, start=1):
                params_str = '; '.join([f"{k}: {v}" for k, v in result['params'].items()])
                mean_mse = result['val_mse']
                rmse_val = np.sqrt(mean_mse)
                line = f"{rank}, {params_str}, {mean_mse:.6f}, {rmse_val:.6f}\n"
                f.write(line)
        print(f"All tested parameters and their MSEs have been logged to {mse_file}.")
        print(f"MSE and RMSE saved to {mse_file}.")
    else:
        print(f"Invalid forecasting type '{forecasting_type}' for {currency}. Skipping predictions.")

def main():
    currency = 'USDJPY=X'
    scaling_method = 'standard'
    forecasting_type = 'recursive'
    combined_data_dir = 'combined_data'
    output_dir = 'forecasting_outputs'
    sequence_length = 14
    dataset_time = 6
    prediction_time = 90
    short_term_lag = 7
    long_term_lag = 30
    param_grid = {
        'rnn_type': ['LSTM'],
        'n_layers': [1],
        'units': [50],
        'activation': [ 'relu'],
        'optimizer': ['adam'],
        'batch_size': [32],
        'epochs': [50]
    }
    os.makedirs(output_dir, exist_ok=True)
    forecast_currency_with_seasonal_data(
        sequence_length, currency, combined_data_dir, output_dir, scaling_method, forecasting_type,
        dataset_time, prediction_time, short_term_lag, long_term_lag, param_grid
    )

if __name__ == "__main__":
    main()
