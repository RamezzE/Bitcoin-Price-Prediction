from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import pandas as pd

def cross_val_time_series(model, X, y, tss):
    preds = []
    y_tests = []
    for train_idx, val_idx in tss.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[val_idx]
        
        y_train = y.iloc[train_idx]
        y_test = y.iloc[val_idx]
          
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        preds.append(y_pred)
        y_tests.append(y_test)
        
    return preds, y_tests
