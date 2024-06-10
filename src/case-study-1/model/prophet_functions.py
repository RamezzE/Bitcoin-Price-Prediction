from prophet import Prophet
import pandas as pd

import logging
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    

logging.getLogger("cmdstanpy").disabled = True

def prophet_train_test(train, test):
    
    prophet_model = Prophet()
    
    train = train.copy()
    test = test.copy()   

    train.rename(columns = {'Date':'ds'}, inplace = True)
    train.rename(columns = {'Close':'y'}, inplace = True)

    test.rename(columns = {'Date':'ds'}, inplace = True)
    test.rename(columns = {'Close':'y'}, inplace = True)

    for col in train.columns:
        if col == 'ds' or col == 'y':
            continue
        
        prophet_model.add_regressor(col)
        
    prophet_model.fit(train)

    forecast = prophet_model.predict(test)

    predictions = forecast['yhat'].values

    return predictions

# Cross Validation for Prophet
def prophet_cross_val(X, y, tss):
    preds = []
    y_tests = []
    
    for train_idx, val_idx in tss.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[val_idx]
        
        y_train = y.iloc[train_idx]
        y_test = y.iloc[val_idx]
        
        # Reset indexes
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        
        train = pd.concat([X_train.copy(), y_train.copy()], axis=1)
        
        # Rename columns for Prophet
        train.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        X_test = X_test.copy()
        X_test.rename(columns={'Date': 'ds'}, inplace=True)
        
        model = Prophet()
        
        for col in train.columns:
            if col != 'ds' and col != 'y':
                model.add_regressor(col)
        
        model.fit(train)
        
        test = X_test.copy()
        
        forecast = model.predict(test)
        y_pred = forecast['yhat'].values
        
        preds.append(y_pred)
        y_tests.append(y_test.values)
        
    return preds, y_tests

def prophet_walk_forward_val(train, test):
    
    all_forecasts = []
    
    train = train.copy()
    test = test.copy()

    train.rename(columns = {'Date':'ds'}, inplace = True)
    train.rename(columns = {'Close':'y'}, inplace = True)

    test.rename(columns = {'Date':'ds'}, inplace = True)
    test.rename(columns = {'Close':'y'}, inplace = True)

    history = train.copy()

    for i in range(len(test)):
        # if i % 10 == 0:
        #     print(f"Iteration {i}/{test.shape[0]}")
        
        prophet_model = Prophet()
        
        # Add regressors to the model
        for col in train.columns:
            if col == 'ds' or col == 'y':
                continue
            
            prophet_model.add_regressor(col)
        
        prophet_model.fit(history)
        
        future = pd.DataFrame(test.iloc[i]).T
        
        forecast = prophet_model.predict(future)
        
        all_forecasts.append(forecast.iloc[0]['yhat'])
        
        next_observation = pd.DataFrame(test.iloc[i]).T 
        history = pd.concat([history, next_observation], ignore_index=True)
        
    return all_forecasts
