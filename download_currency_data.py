import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def download_currency_data_ticker(pairs, start_date, end_date, interval='1d', save_dir='currency_data', add_day_of_week=False):
    for pair in pairs:
        try:
            ticker = yf.Ticker(pair)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            if data.empty:
                continue
            data.reset_index(inplace=True)
            if len(data.columns) > 0:
                first_col = data.columns[0]
                data.rename(columns={first_col: 'Date'}, inplace=True)
            if 'Date' not in data.columns:
                continue
            data = data.dropna(subset=['Date'])
            data = data[~data['Date'].astype(str).str.contains(pair)]
            if not pd.api.types.is_datetime64_any_dtype(data['Date']):
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data.dropna(subset=['Date'])
            data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
            if add_day_of_week:
                data['Day_of_week'] = pd.to_datetime(data['Date']).dt.day_name()
            data['Year'] = pd.to_datetime(data['Date']).dt.year
            data['Month'] = pd.to_datetime(data['Date']).dt.month
            new_columns = []
            for col in data.columns:
                if col in ['Date', 'Day_of_week', 'Year', 'Month']:
                    new_columns.append(col)
                elif col.startswith(('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume')):
                    base_col = col.split('_')[0]
                    new_columns.append(base_col)
                else:
                    new_columns.append(col)
            data.columns = new_columns
            if add_day_of_week:
                required_columns = ['Date', 'Year', 'Month', 'Day_of_week', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            else:
                required_columns = ['Date', 'Year', 'Month', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            available_columns = [col for col in required_columns if col in data.columns]
            if not available_columns:
                continue
            data = data[available_columns]
            file_name = os.path.join(save_dir, f"{pair}.csv")
            data.to_csv(file_name, index=False)
        except Exception as e:
            print(f"Failed to download data for {pair}: {e}")

if __name__ == "__main__":
    currency_pairs = ['USDJPY=X', 'USDEUR=X', 'USDPLN=X', 'USDAUD=X']
    end_date = datetime.today()
    start_date_str = '2010-01-01'
    end_date_str = end_date.strftime('%Y-%m-%d')
    save_directory = 'currency_data'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    download_currency_data_ticker(
        currency_pairs,
        start_date_str,
        end_date_str,
        interval='1d',
        save_dir=save_directory,
        add_day_of_week=True
    )
