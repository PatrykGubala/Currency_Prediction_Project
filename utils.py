import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.feature_selection import SelectKBest, f_regression
from plotting_utils import decompose_time_series
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

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

def filter_last_n_years(data, n_years=6):
    latest_date = data.index.max()
    start_date = latest_date - relativedelta(years=n_years)
    filtered_data = data.loc[start_date:latest_date].copy()
    return filtered_data

def create_features_filtered(data, currency, use_short_term_lag=True, use_long_term_lag=True, short_term_lag=60, long_term_lag=360):
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

def create_seasonal_features_filtered(data, currency, use_short_term_lag=True, use_long_term_lag=True, short_term_lag=60, long_term_lag=360):
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
    feature_cols += ['Month', 'Day_of_Week', 'Quarter', 'Sin_Month', 'Cos_Month', 'Entropy', 'Random_Component', 'OU_Simulated']
    gdp_features = [col for col in data.columns if col.startswith('GDP_Growth_Percentage')]
    if gdp_features:
        feature_cols += gdp_features
    data = data.dropna()
    X = data[feature_cols].copy()
    y = data[close_col].copy()
    return X, y

def select_features(X, y, k=10):
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    return X[selected_features]

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


def prepare_seasonal_data(data, currency, output_dir):
    decomposition = decompose_time_series(data, currency, output_dir)
    data = create_seasonal_features(data, currency)
    data = add_stochastic_features(data, currency)
    data = apply_seasonal_adjustment(data, decomposition, currency)
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
