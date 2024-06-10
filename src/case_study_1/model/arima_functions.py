from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import logging
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

def arimax_train_test(X_train, X_test, y_train, y_test, order = (2,1,0)):

    # Set the datetime index for train and test sets
    X_train = X_train.set_index('Date')
    X_test = X_test.set_index('Date')
    
    # Ensure the index has a frequency
    X_train.index = pd.DatetimeIndex(X_train.index).to_period('D')
    X_test.index = pd.DatetimeIndex(X_test.index).to_period('D')
    
    # Set the datetime index for y_train and y_test
    y_train.index = X_train.index
    y_test.index = X_test.index


    exog_train = X_train
    exog_test = X_test

    arima_model = SARIMAX(y_train, exog=exog_train, order = order)

    arima_model_fit = arima_model.fit(disp=False)

    forecast = arima_model_fit.get_forecast(steps=len(y_test), exog=exog_test)

    predictions = forecast.predicted_mean
    
    return predictions

def arimax_cross_val(X, y, tss, order = (2,1,0)):

    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    
    preds = []
    y_tests = []
    
    for train_idx, val_idx in tss.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[val_idx]
        
        y_train = y.iloc[train_idx]
        y_test = y.iloc[val_idx]
        
        model = SARIMAX(y_train, exog=X_train, order=order)
        model_fit = model.fit(disp=False)
                
        forecast = model_fit.get_forecast(steps=len(y_test), exog = X_test)
        y_pred = forecast.predicted_mean

        preds.append(y_pred)
        y_tests.append(y_test)
        
    return preds, y_tests

def arima_walk_forward_val(y_train, y_test, order = (2,1,0)):
    predictions = []

    train = [x for x in y_train]
    test = [x for x in y_test]

    history = train.copy()

    for i in range(len(test)):

        model = ARIMA(history, order = order)
        model_fit = model.fit()

        output = model_fit.forecast()

        y_hat = output[0]

        predictions.append(y_hat)

        history.append(test[i])
        
    return predictions