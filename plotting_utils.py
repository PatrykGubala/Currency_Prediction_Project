import matplotlib.pyplot as plt
import os
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_line_graph(x_data_list, y_data_list, labels, title, x_label, y_label, legend_labels, output_path, figure_size=(14,7)):
    plt.figure(figsize=figure_size)
    for x_data, y_data, label in zip(x_data_list, y_data_list, legend_labels):
        plt.plot(x_data, y_data, label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
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
        figure_size=(14, 7)
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
        figure_size=(14, 7)
    )


def plot_training_history(history, output_dir, currency):
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    if not loss:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Training Loss')
    if val_loss:
        plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{currency} Training History')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{currency}_training_history.png')
    plt.savefig(plot_path)
    plt.close()