import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout

from plotting_utils import plot_training_loss_by_rnn_type, plot_validation_loss_by_rnn_type, plot_residuals_histogram, \
    plot_residuals_over_time, plot_scatter_actual_vs_predicted, plot_results, plot_heatmap, plot_line_graph


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
        figure_size=(12, 10),
        annotate=True

    )




def add_stochastic_features(data, currency):
    close_col = f'Close_{currency}'
    def calculate_entropy(series):
        probabilities = series / series.sum()
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    data['Entropy'] = calculate_entropy(data[close_col])

    data['Random_Component'] = np.random.normal(0, data[close_col].std(), len(data))

    def ornstein_uhlenbeck_process(mu, sigma, theta, T, N):
        dt = T / N
        process = np.zeros(N)
        process[0] = mu
        for t in range(1, N):
            process[t] = process[t - 1] + theta * (mu - process[t - 1]) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1)
        return process
    data['OU_Simulated'] = ornstein_uhlenbeck_process(
        mu=data[close_col].mean(),
        sigma=data[close_col].std(),
        theta=0.1,
        T=1,
        N=len(data)
    )
    return data

def create_seasonal_features(data, currency):
    close_col = f'Close_{currency}'
    data['Month'] = data.index.month
    data['Day_of_Week'] = data.index.dayofweek
    data['Quarter'] = data.index.quarter
    data['Sin_Month'] = np.sin(data['Month'] * (2 * np.pi / 12))
    data['Cos_Month'] = np.cos(data['Month'] * (2 * np.pi / 12))
    return data

def apply_seasonal_adjustment(data, decomposition_result, currency):
    close_col = f'Close_{currency}'
    adjusted_data = data.copy()
    adjusted_data['Seasonal_Adjusted_Close'] = data[close_col] / decomposition_result.seasonal
    return adjusted_data

def prepare_seasonal_data(data, currency, output_dir):
    decomposition = decompose_time_series(data, currency, output_dir)
    data = create_seasonal_features(data, currency)
    data = add_stochastic_features(data, currency)
    data = apply_seasonal_adjustment(data, decomposition, currency)
    return data





def create_lstm_sequences(X, y, sequence_length=1):
    X_seq = []
    y_seq = []
    for i in range(len(X) - sequence_length):
        X_seq.append(X.iloc[i:i + sequence_length].values)
        y_seq.append(y.iloc[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)


def make_predictions(best_model, X_test_seq, y_test_seq, scaler_y):
    predictions_scaled = best_model.predict(X_test_seq)
    predictions_scaled = pd.Series(predictions_scaled.flatten()).rolling(window=5, min_periods=1).mean().values
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    y_test_plot = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    return predictions, y_test_plot

def create_rnn_model(rnn_type='LSTM', n_layers=1, units=50, activation='tanh', optimizer='adam', input_shape=None, dropout=0.0):
    if input_shape is None:
        raise ValueError("input_shape must be specified")
    model = Sequential()
    for i in range(n_layers):
        return_sequences = True if i < n_layers - 1 else False
        if rnn_type == 'LSTM':
            model.add(
                LSTM(units=units, activation=activation, input_shape=input_shape, return_sequences=return_sequences, dropout=dropout)
            )
        elif rnn_type == 'GRU':
            model.add(
                GRU(units=units, activation=activation, input_shape=input_shape, return_sequences=return_sequences, dropout=dropout)
            )
        else:
            model.add(
                SimpleRNN(units=units, activation=activation, input_shape=input_shape, return_sequences=return_sequences, dropout=dropout)
            )
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    return model

def train_rnn_models(X_train_seq, y_train_seq, X_val_seq, y_val_seq, sequence_length, param_grid, model_counter, total_models):
    results = []
    histories = []
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    param_combinations = list(product(*values))
    for params in param_combinations:
        param_dict = dict(zip(keys, params))
        model_counter.increment()
        print("")
        print("ARGUMENTS TABLE:")
        print(pd.DataFrame({'Parameter': list(param_dict.keys()), 'Value': list(param_dict.values())}))
        model = create_rnn_model(
            rnn_type=param_dict['rnn_type'],
            n_layers=param_dict['n_layers'],
            units=param_dict['units'],
            activation=param_dict['activation'],
            optimizer=param_dict['optimizer'],
            input_shape=(sequence_length, X_train_seq.shape[2]),
            dropout=param_dict['dropout']
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=param_dict['epochs'],
            batch_size=param_dict['batch_size'],
            callbacks=[early_stopping],
            verbose=1
        )
        val_loss = min(history.history['val_loss'])
        val_preds = model.predict(X_val_seq)
        val_r2 = r2_score(y_val_seq, val_preds)
        results.append({
            'params': param_dict,
            'val_loss': val_loss,
            'val_r2': val_r2,
            'history': history
        })
        histories.append(history)
    return results, histories, param_combinations

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

def feature_selection(X, y, k_best_features=10):
    selector = SelectKBest(f_regression, k=k_best_features)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    return pd.DataFrame(X_new, columns=selected_features, index=X.index), selected_features

def filter_last_n_years(data, n_years=6):
    if not isinstance(data.index, pd.DatetimeIndex):
        return data
    end_date = data.index.max()
    start_date = end_date - pd.DateOffset(years=n_years)
    filtered_data = data.loc[(data.index >= start_date) & (data.index <= end_date)]
    return filtered_data



def create_seasonal_features_filtered(data, currency, use_short_term_lag=True, use_long_term_lag=True, short_term_lag=60, long_term_lag=360):
    data = data.copy()
    close_col = f'Close_{currency}'
    if close_col not in data.columns:
        return pd.DataFrame(), pd.Series()

    if use_short_term_lag:
        data[f'Close_Lag_{short_term_lag}'] = data[close_col].shift(short_term_lag)
    if use_long_term_lag:
        data[f'Close_Lag_{long_term_lag}'] = data[close_col].shift(long_term_lag)
    data['MA_5'] = data[close_col].rolling(window=5).mean()

    expected_cols = [
        'Close_Lag_'+str(short_term_lag),
        'Close_Lag_'+str(long_term_lag),
        'MA_5',
        'Month',
        'Day_of_Week',
        'Quarter',
        'Sin_Month',
        'Cos_Month',
        'Entropy',
        'Random_Component',
        'OU_Simulated',
        'Seasonal_Adjusted_Close'
    ]
    gdp_features = [col for col in data.columns if col.startswith('GDP_Growth_Percentage')]
    expected_cols += gdp_features

    data = data.dropna(subset=[col for col in expected_cols if col in data.columns])
    X = data[[col for col in expected_cols if col in data.columns]].copy()
    y = data[close_col].copy()
    return X, y


def forecast_currency_with_seasonal_data(
    sequence_length,
    currency,
    combined_data_dir,
    output_dir,
    scaling_method='standard',
    forecasting_type='recursive',
    dataset_time=6,
    prediction_time=180,
    short_term_lag=60,
    long_term_lag=360,
    param_grid=None,
    k_best_features=10
):
    currency_output_dir = os.path.join(output_dir, currency)
    os.makedirs(currency_output_dir, exist_ok=True)
    file_name = f'combined_data_{currency}.csv'
    file_path = os.path.join(combined_data_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Combined data file for {currency} not found at {file_path}.")
        return
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    if data is None or data.empty:
        print(f"No data available for {currency}.")
        return

    data = data.drop(columns=['Day_of_Week'], errors='ignore')
    data = prepare_seasonal_data(data, currency, output_dir)

    filtered_data = filter_last_n_years(data, n_years=dataset_time)
    if filtered_data.empty:
        print(f"Filtered data for {currency} is empty.")
        return

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
        return

    string_columns = X.select_dtypes(include=['object', 'string']).columns
    if len(string_columns) > 0:
        print("Usuwam kolumny tekstowe:", string_columns)
        X = X.drop(columns=string_columns, errors='ignore')

    print("")
    print("FEATURES TABLE (BEFORE FEATURE SELECTION):")
    print(pd.DataFrame({'Feature': X.columns}))

    X_fs, selected_features = feature_selection(X, y, k_best_features=k_best_features)

    print("")
    print("FEATURES TABLE (AFTER FEATURE SELECTION):")
    print(pd.DataFrame({'Feature': selected_features}))

    y = y.loc[X_fs.index]

    visualize_data(pd.concat([X_fs, y.rename('Close_' + currency)], axis=1), currency, currency_output_dir)

    if forecasting_type == 'recursive':
        test_period_days = prediction_time
        train_size = len(X_fs) - test_period_days - sequence_length
        if train_size <= 0:
            print(f"Not enough data to train for {currency}.")
            return

        X_train_df = X_fs.iloc[:train_size + sequence_length]
        X_val_df = X_fs.iloc[train_size:]
        y_train_series = y.iloc[:train_size + sequence_length]
        y_val_series = y.iloc[train_size:]

        scaler_X.fit(X_train_df)
        X_train_scaled = scaler_X.transform(X_train_df)
        X_val_scaled = scaler_X.transform(X_val_df)

        scaler_y.fit(y_train_series.values.reshape(-1, 1))
        y_train_scaled = scaler_y.transform(y_train_series.values.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val_series.values.reshape(-1, 1)).flatten()

        X_train_seq, y_train_seq = create_lstm_sequences(
            pd.DataFrame(X_train_scaled, columns=selected_features, index=X_train_df.index),
            pd.Series(y_train_scaled, index=y_train_series.index),
            sequence_length
        )
        X_val_seq, y_val_seq = create_lstm_sequences(
            pd.DataFrame(X_val_scaled, columns=selected_features, index=X_val_df.index),
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
        results, histories, param_combinations = train_rnn_models(
            X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            sequence_length, param_grid, model_counter, total_models
        )

        plot_training_loss_by_rnn_type(results, currency, currency_output_dir)
        plot_validation_loss_by_rnn_type(results, currency, currency_output_dir)

        best_per_rnn = {}
        for rnn_type in ['LSTM', 'GRU', 'SimpleRNN']:
            filtered = [res for res in results if res['params']['rnn_type'] == rnn_type]
            if not filtered:
                continue
            best_result_rnn = min(filtered, key=lambda x: x['val_loss'])
            best_per_rnn[rnn_type] = best_result_rnn

        print("BEST MODELS PER RNN TYPE:")
        for rnn_type, best_res in best_per_rnn.items():
            print(
                f"- {rnn_type}: Params={best_res['params']}, val_loss={best_res['val_loss']}, val_r2={best_res['val_r2']}")
        print(f"Completed manual grid search for {currency}.")

        best_result = min(results, key=lambda x: x['val_loss'])
        best_params = best_result['params']
        best_history = best_result['history']

        print(f"Best Parameters for {currency} RNN: {best_params}")
        print(f"Best Validation MSE (loss) for {currency} RNN: {best_result['val_loss']}")

        best_model = create_rnn_model(
            rnn_type=best_params['rnn_type'],
            n_layers=best_params['n_layers'],
            units=best_params['units'],
            activation=best_params['activation'],
            optimizer=best_params['optimizer'],
            input_shape=(sequence_length, X_train_seq.shape[2]),
            dropout=best_params['dropout']
        )
        best_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)],
            verbose=0
        )

        predictions, y_val_plot = make_predictions(best_model, X_val_seq, y_val_seq, scaler_y)
        y_test_dates = y_val_series.index[sequence_length:]
        assert len(y_val_seq) == len(predictions) == len(y_test_dates), "Lengths of test data do not match."

        residuals = y_val_plot - predictions

        hist_path = os.path.join(currency_output_dir, f'{currency}_residuals_hist.png')
        time_path = os.path.join(currency_output_dir, f'{currency}_residuals_over_time.png')
        scatter_path = os.path.join(currency_output_dir, f'{currency}_actual_vs_predicted_scatter.png')

        plot_residuals_histogram(residuals, hist_path)
        plot_residuals_over_time(y_test_dates, residuals, time_path)
        plot_scatter_actual_vs_predicted(y_val_plot, predictions, scatter_path)


        plot_results(
            currency,
            y_train_seq,
            y_val_plot,
            predictions,
            y_train_series.index[sequence_length:],
            y_test_dates,
            scaler_y,
            currency_output_dir,
            dataset_time
        )

        mse = mean_squared_error(y_val_plot, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val_plot, predictions)
        print(f'Mean Squared Error for {currency}: {mse}')
        print(f'Root Mean Squared Error for {currency}: {rmse}')
        print(f'R2 Score for {currency}: {r2}')

        mse_file_csv = os.path.join(currency_output_dir, f'{currency}_mse_results.csv')
        sorted_results = sorted(results, key=lambda x: x['val_loss'])
        rows = []
        rank = 1
        for result in sorted_results:
            params_str = '; '.join([f"{k}: {v}" for k, v in result['params'].items()])
            mean_mse = result['val_loss']
            rmse_val = np.sqrt(mean_mse)
            val_r2 = result['val_r2']
            rows.append([rank, params_str, mean_mse, rmse_val, val_r2])
            rank += 1
        df_results = pd.DataFrame(rows, columns=["Rank", "Parameters", "Mean MSE", "RMSE", "R2"])
        df_results.to_csv(mse_file_csv, index=False)
        print(f"All tested parameters and their MSEs have been logged to {mse_file_csv}.")
    else:
        print(f"Invalid forecasting type '{forecasting_type}' for {currency}. Skipping predictions.")


def main():
    currencies = ['USDPLN=X', 'USDEUR=X']

    scaling_method = 'standard'
    forecasting_type = 'recursive'
    combined_data_dir = 'combined_data'
    output_dir = 'forecasting_outputs'
    sequence_length = 14
    dataset_time = 6
    prediction_time = 30
    short_term_lag = 7
    long_term_lag = 30
    param_grid = {
        'rnn_type': ['LSTM', 'GRU', 'SimpleRNN'],
        'n_layers': [1, 2],
        'units': [50],
        'activation': ['relu', 'tanh'],
        'optimizer': ['adam'],
        'batch_size': [32],
        'epochs': [20, 50],
        'dropout': [0.0]
    }
    os.makedirs(output_dir, exist_ok=True)
    for currency in currencies:
        forecast_currency_with_seasonal_data(
            sequence_length,
            currency,
            combined_data_dir,
            output_dir,
            scaling_method,
            forecasting_type,
            dataset_time,
            prediction_time,
            short_term_lag,
            long_term_lag,
            param_grid,
            k_best_features=8
        )

if __name__ == "__main__":
    main()
