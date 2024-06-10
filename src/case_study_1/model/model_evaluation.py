import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(model_name, dates, test_data, predictions, plot = True):
    '''
    This function evaluates the model performance by calculating
    MAE, MSE, RMSE, R2, and MAPE.
    '''
    
    mae = mean_absolute_error(test_data, predictions)
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_data, predictions)
    mape = mean_absolute_percentage_error(test_data, predictions)
    
    print(f'{model_name}:\nMAE: {mae:.3f} -- MSE: {mse:.3f} -- RMSE: {rmse:.3f}')
    print(f'R2: {r2:.3f} -- MAPE: {mape:.3f}')
    
    if plot:
        plot_results(model_name, dates, test_data, predictions)


def plot_results(model_name, dates, test_data, predictions):

  plt.figure(figsize=(10, 6))

  plt.plot(dates, test_data, label='Actual', color='blue')

  plt.plot(dates, predictions, label='Predicted', color='red')

  plt.xlabel('Date')
  plt.ylabel('Value')
  plt.title(f'Actual vs Predicted Values ({model_name})')
  plt.legend()
  plt.show()
  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np

def evaluate_cross_val(preds, y_tests):
    mae_scores = []
    mse_scores = []
    r2_scores = []
    mape_scores = []

    for y_pred, y_test in zip(preds, y_tests):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        mae_scores.append(mae)
        mse_scores.append(mse)
        r2_scores.append(r2)
        mape_scores.append(mape)
                
    mae = np.mean(mae_scores)
    mse = np.mean(mse_scores)
    r2 = np.mean(r2_scores)
    mape = np.mean(mape_scores)
        
    print(f"R2: {r2:.3f} -- MSE: {mse:.3f} -- MAE: {mae:.3f} -- MAPE: {mape:.3f}")
    
