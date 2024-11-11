import pandas as pd
import os
from datetime import datetime

def process_currency(currency, currency_dir, gdp_dir, output_dir, start_date):

    date_range = pd.date_range(start=start_date, end=datetime.today(), freq='D')

    currency_file = f'{currency}.csv'
    currency_path = os.path.join(currency_dir, currency_file)
    if not os.path.exists(currency_path):
        print(f"Currency data file not found for {currency} at {currency_path}. Skipping.")
        return

    try:
        currency_data = pd.read_csv(currency_path)
    except Exception as e:
        print(f"Error reading currency data for {currency}: {e}")
        return

    if 'Date' not in currency_data.columns:
        print(f"'Date' column missing in currency data for {currency}. Skipping.")
        return

    currency_data['Date'] = pd.to_datetime(currency_data['Date'])
    currency_data = currency_data.drop(columns=['Year', 'Month', 'Day_of_week'], errors='ignore')
    currency_data = currency_data.set_index('Date').reindex(date_range).ffill().reset_index()
    currency_data = currency_data.rename(columns={
        'index': 'Date',
        'Open': f'Open_{currency}',
        'High': f'High_{currency}',
        'Low': f'Low_{currency}',
        'Close': f'Close_{currency}',
        'Volume': f'Volume_{currency}'
    })

    gdp_file = f'gdp_data_{currency}.csv'
    gdp_path = os.path.join(gdp_dir, gdp_file)
    if not os.path.exists(gdp_path):
        print(f"GDP data file not found for {currency} at {gdp_path}. Skipping.")
        return

    try:
        gdp_data = pd.read_csv(gdp_path)
    except Exception as e:
        print(f"Error reading GDP data for {currency}: {e}")
        return

    if 'Date' not in gdp_data.columns:
        print(f"'Date' column missing in GDP data for {currency}. Skipping.")
        return

    gdp_data['Date'] = pd.to_datetime(gdp_data['Date'])
    gdp_data = gdp_data.set_index('Date').reindex(date_range).ffill().reset_index()
    gdp_data = gdp_data.rename(columns={'index': 'Date'})
    gdp_data.sort_values(by='Date', inplace=True)

    combined_data = pd.merge(currency_data, gdp_data, on='Date', how='left')
    combined_data = combined_data.ffill()

    combined_data['Year'] = combined_data['Date'].dt.year
    combined_data['Month'] = combined_data['Date'].dt.month
    combined_data['Quarter'] = combined_data['Date'].dt.quarter
    combined_data['Day_of_week'] = combined_data['Date'].dt.day_name()

    columns_order = ['Date', 'Year', 'Month', 'Quarter', 'Day_of_week'] + \
                   [col for col in combined_data.columns if col not in ['Date', 'Year', 'Month', 'Quarter', 'Day_of_week']]
    combined_data = combined_data[columns_order]
    combined_data.sort_values('Date', inplace=True)

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'combined_data_{currency}.csv')
    try:
        combined_data.to_csv(output_file, index=False)
        print(f"Combined data for {currency} saved to {output_file}.")
    except Exception as e:
        print(f"Error saving combined data for {currency}: {e}")

def combine_data(currency_dir='currency_data',
                gdp_dir='gdp_data',
                output_dir='combined_data',
                start_date='2010-01-01'):

    currencies = ['USDPLN=X', 'USDAUD=X', 'USDEUR=X', 'USDJPY=X']
    for currency in currencies:
        print(f"\nProcessing currency pair: {currency}")
        process_currency(currency, currency_dir, gdp_dir, output_dir, start_date)

if __name__ == "__main__":
    combine_data()
