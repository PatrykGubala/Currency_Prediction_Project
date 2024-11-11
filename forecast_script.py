import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import os
import warnings

from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def plot_line_graph(x_data_list, y_data_list, labels, title, x_label, y_label, legend_labels, output_path, figure_size=(14,7)):
    plt.figure(figsize=figure_size)
    for x_data, y_data, label in zip(x_data_list, y_data_list, legend_labels):
        plt.plot(x_data, y_data, label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
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
    plt.savefig(output_path)
    plt.close()

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None
    if 'Date' not in data.columns:
        print(f"'Date' column missing in {file_path}.")
        return None
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True)
    return data

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

def create_features_filtered(data, currency, use_short_term_lag=True, use_long_term_lag=True, short_term_lag=60,
                             long_term_lag=360):
    close_col = f'Close_{currency}'
    feature_cols = []
    if use_short_term_lag:
        data[f'Close_Lag_{short_term_lag}'] = data[close_col].shift(short_term_lag)
        feature_cols.append(f'Close_Lag_{short_term_lag}')
    if use_long_term_lag:
        data[f'Close_Lag_{long_term_lag}'] = data[close_col].shift(long_term_lag)
        feature_cols.append(f'Close_Lag_{long_term_lag}')
    data['MA_5'] = data[close_col].rolling(window=5).mean()
    feature_cols += ['MA_5']
    gdp_features = [col for col in data.columns if col.startswith('GDP_Growth_Percentage')]
    if gdp_features:
        feature_cols += gdp_features
    data = data.dropna()
    X = data[feature_cols].copy()
    y = data[close_col].copy()
    return X, y

def create_lstm_sequences(X, y, sequence_length=1):
    X_seq = []
    y_seq = []
    for i in range(len(X) - sequence_length):
        X_seq.append(X.iloc[i:i+sequence_length].values)
        y_seq.append(y.iloc[i+sequence_length])
    return np.array(X_seq), np.array(y_seq)

def filter_last_n_years(data, n_years=6):
    latest_date = data.index.max()
    start_date = latest_date - relativedelta(years=n_years)
    filtered_data = data.loc[start_date:latest_date].copy()
    return filtered_data

def forecast_currency(currency, combined_data_dir, output_dir, scaling_method='standard', forecasting_type='direct',
                      dataset_time=6, prediction_time=180, short_term_lag=60, long_term_lag=360):
    file_name = f'combined_data_{currency}.csv'
    file_path = os.path.join(combined_data_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Combined data file for {currency} not found at {file_path}. Skipping.")
        return
    currency_output_dir = os.path.join(output_dir, currency)
    os.makedirs(currency_output_dir, exist_ok=True)
    data = load_data(file_path)
    if data is None or data.empty:
        print(f"No data available for {currency}. Skipping.")
        return
    data = data.drop(columns=['Day_of_week'], errors='ignore')
    visualize_data(data, currency, currency_output_dir)
    filtered_data = filter_last_n_years(data, n_years=dataset_time)
    if filtered_data.empty:
        print(f"Filtered data for {currency} is empty. Skipping.")
        return
    print(f"Filtered Data Range for {currency}: {filtered_data.index.min().date()} to {filtered_data.index.max().date()}")
    X, y = create_features_filtered(
        filtered_data,
        currency=currency,
        use_short_term_lag=True,
        use_long_term_lag=True,
        short_term_lag=short_term_lag,
        long_term_lag=long_term_lag
    )
    if X.empty or y.empty:
        print(f"No features/target available after filtering for {currency}. Skipping.")
        return
    feature_columns = X.columns.tolist()
    scaler_options = {
        'standard': StandardScaler(),
        'normalize': MinMaxScaler(),
        'robust': RobustScaler()
    }
    scaler = scaler_options.get(scaling_method, StandardScaler())
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
    if forecasting_type == 'recursive':
        sequence_length = 10
        test_period_days = prediction_time
        train_size = len(X_scaled_df) - test_period_days - sequence_length
        if train_size <= 0:
            print(f"Not enough data to train for {currency}. Skipping.")
            return
        X_train_df = X_scaled_df.iloc[:train_size + sequence_length]
        X_test_df = X_scaled_df.iloc[train_size:]
        y_train_series = y.iloc[:train_size + sequence_length]
        y_test_series = y.iloc[train_size:]
        X_train_seq, y_train_seq = create_lstm_sequences(X_train_df, y_train_series, sequence_length)
        X_test_seq, y_test_seq = create_lstm_sequences(X_test_df, y_test_series, sequence_length)
        if len(X_test_seq) == 0:
            print(f"Not enough test sequences after adjustment for {currency}. Skipping.")
            return
        print(f"Training set size for {currency}: {X_train_seq.shape[0]}")
        print(f"Testing set size for {currency}: {X_test_seq.shape[0]}")
        def create_lstm_model(units=50, activation='tanh', optimizer='adam'):
            model = Sequential()
            model.add(Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
            model.add(LSTM(units=units, activation=activation))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer=optimizer)
            return model
        model = KerasRegressor(model=create_lstm_model, verbose=1)
        param_grid = {
            'model__units': [50, 100],
            'model__activation': ['relu'],
            'optimizer': ['adam'],
            'batch_size': [16],
            'epochs': [100]
        }
        tscv = TimeSeriesSplit(n_splits=3)
        warnings.filterwarnings('ignore')
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=tscv,
            verbose=1,
            n_jobs=-1
        )
        print(f"Starting GridSearchCV for {currency} (Recursive Forecasting with LSTM)...")
        grid_search.fit(X_train_seq, y_train_seq)
        print(f"Completed GridSearchCV for {currency}.")
        best_model = grid_search.best_estimator_
        print(f"Best Parameters for {currency} LSTM: {grid_search.best_params_}")
        print(f"Best Cross-Validation Score for {currency} LSTM: {grid_search.best_score_}")
        predictions = best_model.predict(X_test_seq)
        predictions = pd.Series(predictions).rolling(window=5, min_periods=1).mean().values
        y_train_plot = y.iloc[sequence_length:train_size + sequence_length]
        y_test_plot = y.iloc[train_size + sequence_length:]
        test_dates = y_test_plot.index
        assert len(y_test_seq) == len(predictions) == len(test_dates), "Lengths of test data do not match."
        forecast_plot_path = os.path.join(currency_output_dir, f'{currency}_forecast.png')
        plot_line_graph(
            x_data_list=[y_train_plot.index, test_dates, test_dates],
            y_data_list=[y_train_seq, y_test_seq, predictions],
            labels=['Training Data', 'Actual Prices', 'Predicted Prices'],
            title=f'{currency} Closing Price Prediction (Last {dataset_time} Years)',
            x_label='Date',
            y_label='Price',
            legend_labels=['Training Data', 'Actual Prices', 'Predicted Prices'],
            output_path=forecast_plot_path,
            figure_size=(14, 7)
        )
        comparison_plot_path = os.path.join(currency_output_dir, f'{currency}_actual_vs_predicted.png')
        plot_line_graph(
            x_data_list=[test_dates, test_dates],
            y_data_list=[y_test_seq, predictions],
            labels=['Actual Prices', 'Predicted Prices'],
            title=f'{currency} Actual vs Predicted Prices (Test Set)',
            x_label='Date',
            y_label='Price',
            legend_labels=['Actual Prices', 'Predicted Prices'],
            output_path=comparison_plot_path,
            figure_size=(14, 7)
        )
        mse = mean_squared_error(y_test_seq, predictions)
        print(f'Mean Squared Error for {currency}: {mse}')
        mse_file = os.path.join(currency_output_dir, f'{currency}_mse.txt')
        with open(mse_file, 'w') as f:
            f.write(f'Mean Squared Error: {mse}\n')
        print(f"MSE saved to {mse_file}.")
    else:
        print(f"Invalid forecasting type '{forecasting_type}' for {currency}. Skipping predictions.")
        return

def main():
    currency = 'USDPLN=X'
    scaling_method = 'standard'
    forecasting_type = 'recursive'
    combined_data_dir = 'combined_data'
    output_dir = 'forecasting_outputs'
    dataset_time = 6
    prediction_time = 180
    short_term_lag = 7
    long_term_lag = 90
    os.makedirs(output_dir, exist_ok=True)
    forecast_currency(currency, combined_data_dir, output_dir, scaling_method, forecasting_type,
                      dataset_time, prediction_time, short_term_lag, long_term_lag)

if __name__ == "__main__":
    main()
