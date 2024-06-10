import matplotlib.pyplot as plt
import os

def plot_all_results(model_names, dates, test_data, predictions, title, save_path):
    """
    Plots the actual and predicted values and saves the plot to the specified path.

    Parameters:
    model_names (list): List of model names.
    dates (list): List of dates corresponding to the data points.
    test_data (list): List of actual test data values.
    predictions (list of lists): List of lists containing predictions from different models.
    title (str): Title of the plot.
    save_path (str): Path to save the plot.
    """
    
    plt.figure(figsize=(10, 6))

    
    plt.plot(dates, test_data, label='Actual', color='blue', linewidth=2)

    if len(model_names) != len(predictions):
        raise ValueError("The length of model_names and predictions must be the same.")

    colors = ['red', 'green', 'orange', 'purple', 'gray', 'brown', 'pink', 'cyan']

    for model_name, prediction, color in zip(model_names, predictions, colors):
        plt.plot(dates, prediction, label=model_name, color=color)

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)

    plt.show()