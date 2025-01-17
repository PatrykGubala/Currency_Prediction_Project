import matplotlib.pyplot as plt
import os
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose




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

def plot_training_loss_by_rnn_type(results, currency, output_dir):
    rnn_types = ['LSTM', 'GRU', 'SimpleRNN']
    for rnn_type in rnn_types:
        filtered_results = [res for res in results if res['params']['rnn_type'] == rnn_type]
        if not filtered_results:
            continue
        plt.figure(figsize=(25, 8))
        plt.title(f'{currency} - {rnn_type} Training Loss')
        for res in filtered_results:
            history = res['history']
            epochs = range(1, len(history.history['loss']) + 1)
            plt.plot(epochs, history.history['loss'], label=str(res['params']))
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{currency}_{rnn_type}_training_loss.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Training loss plot for {rnn_type} saved to {plot_path}")

def plot_validation_loss_by_rnn_type(results, currency, output_dir):
    rnn_types = ['LSTM', 'GRU', 'SimpleRNN']
    for rnn_type in rnn_types:
        filtered_results = [res for res in results if res['params']['rnn_type'] == rnn_type]
        if not filtered_results:
            continue
        plt.figure(figsize=(25, 8))
        plt.title(f'{currency} - {rnn_type} Validation Loss')
        for res in filtered_results:
            history = res['history']
            epochs = range(1, len(history.history['val_loss']) + 1)
            plt.plot(epochs, history.history['val_loss'], label=str(res['params']))
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{currency}_{rnn_type}_validation_loss.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Validation loss plot for {rnn_type} saved to {plot_path}")

def plot_line_graph(x_data_list, y_data_list, labels, title, x_label, y_label, legend_labels, output_path, figure_size=(14, 7)):
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

def plot_heatmap(data, title, x_tick_labels, y_tick_labels, output_path, figure_size=(12,10), annotate=False):
    plt.figure(figsize=figure_size)
    mat = plt.matshow(data, fignum=1, cmap='coolwarm')
    plt.xticks(range(len(x_tick_labels)), x_tick_labels, rotation=90)
    plt.yticks(range(len(y_tick_labels)), y_tick_labels)
    plt.colorbar(mat)
    if annotate:
        for (i, j), val in np.ndenumerate(data):
            plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
    plt.title(title, pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def decompose_time_series(data, currency, output_dir):
    from statsmodels.tsa.seasonal import seasonal_decompose
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

def plot_residuals_histogram(residuals, output_path, figure_size=(10,6)):
    plt.figure(figsize=figure_size)
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_residuals_over_time(dates, residuals, output_path, figure_size=(14,7)):
    plt.figure(figsize=figure_size)
    plt.plot(dates, residuals, marker='o', linestyle='-', markersize=3)
    plt.title('Residuals over Time')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_scatter_actual_vs_predicted(actual, predicted, output_path, figure_size=(10,6)):
    plt.figure(figsize=figure_size)
    plt.scatter(actual, predicted, alpha=0.5)
    plt.title('Scatter Plot: Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()






