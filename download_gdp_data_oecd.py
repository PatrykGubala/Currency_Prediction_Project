import numpy as np
import requests
import pandas as pd
from datetime import datetime
import os
import xml.etree.ElementTree as ET

def download_gdp_data_oecd(
    url_base="https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD,1.1/Q..",
    countries="USA+EA20+POL+AUS+JPN",
    series="S1..B1GQ......G1.",
    start_period="2010-Q1",
    save_dir='gdp_data',
    file_name_template='gdp_data_{currency}.csv',
    currency_gdp_mapping=None
):
    if currency_gdp_mapping is None:
        currency_gdp_mapping = {
            'USDPLN=X': ['POL'],
            'USDAUD=X': ['AUS'],
            'USDEUR=X': ['EA20'],
            'USDJPY=X': ['JPN']
        }

    try:
        url = f"{url_base}{countries}.{series}?startPeriod={start_period}&dimensionAtObservation=AllDimensions"
        print(f"Downloading GDP data from: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            return

        root = ET.fromstring(response.content)
        namespaces = {
            'message': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message',
            'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic',
            'common': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common'
        }

        records = []
        for obs in root.findall('.//generic:Obs', namespaces):
            obs_key = obs.find('generic:ObsKey', namespaces)
            key_values = {val.attrib['id']: val.attrib['value'] for val in obs_key.findall('generic:Value', namespaces)}
            period = key_values.get('TIME_PERIOD')
            ref_area = key_values.get('REF_AREA')
            value = obs.find('generic:ObsValue', namespaces).attrib.get('value')

            if not value or not period or not ref_area:
                continue

            try:
                year, quarter = period.split('-Q')
                month = int(quarter) * 3 - 2
                date = f"{year}-{month:02d}-01"
                records.append({
                    'Date': date,
                    'Year': int(year),
                    'Quarter': int(quarter),
                    'Country_Code': ref_area,
                    'GDP_Growth_Percentage': float(value)
                })
            except Exception as e:
                print(f"Error processing observation: {e}")
                continue

        if records:
            df = pd.DataFrame(records)
            pivot_df = df.pivot_table(
                index=['Date', 'Year', 'Quarter'],
                columns='Country_Code',
                values='GDP_Growth_Percentage'
            ).reset_index()

            pivot_df.columns = ['Date', 'Year', 'Quarter'] + [f'GDP_Growth_Percentage_{col}' for col in pivot_df.columns if col not in ['Date', 'Year', 'Quarter']]

            pivot_df['Date'] = pd.to_datetime(pivot_df['Date'])

            date_range = pd.date_range(start='2010-01-01', end=datetime.today(), freq='D')

            pivot_df = pivot_df.set_index('Date').reindex(date_range).ffill().reset_index()
            pivot_df.rename(columns={'index': 'Date'}, inplace=True)

            pivot_df.sort_values(by='Date', inplace=True)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for currency, countries_list in currency_gdp_mapping.items():
                if 'USA' not in countries_list:
                    countries_list = ['USA'] + countries_list

                gdp_columns = [f'GDP_Growth_Percentage_{country}' for country in countries_list]

                missing_cols = [col for col in gdp_columns if col not in pivot_df.columns]
                if missing_cols:
                    print(f"Warning: Missing GDP columns {missing_cols} for currency pair {currency}. These will be filled with NaN.")
                    for col in missing_cols:
                        pivot_df[col] = np.nan

                selected_cols = ['Date', 'Year', 'Quarter'] + gdp_columns
                gdp_subset = pivot_df[selected_cols].copy()

                gdp_subset = gdp_subset.ffill()

                gdp_subset.sort_values(by='Date', inplace=True)

                csv_path = os.path.join(save_dir, file_name_template.format(currency=currency))
                gdp_subset.to_csv(csv_path, index=False)
                print(f"GDP data for {currency} successfully saved to {csv_path}.")

        else:
            print("No valid records found in the GDP data.")

    except Exception as e:
        print(f"Failed to download GDP data: {e}")

if __name__ == "__main__":
    currency_gdp_mapping = {
        'USDPLN=X': ['POL'],
        'USDAUD=X': ['AUS'],
        'USDEUR=X': ['EA20'],
        'USDJPY=X': ['JPN']
    }

    download_gdp_data_oecd(
        currency_gdp_mapping=currency_gdp_mapping
    )
