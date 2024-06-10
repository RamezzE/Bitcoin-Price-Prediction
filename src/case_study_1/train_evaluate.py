from preprocessing import preprocess_data, clean_data
import yaml
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from model.cross_validation import cross_val_time_series
from model.model_evaluation import evaluate_cross_val, evaluate_model
from model.arima_functions import arimax_train_test, arimax_cross_val, arima_walk_forward_val
from model.prophet_functions import prophet_train_test, prophet_cross_val, prophet_walk_forward_val
from model.walk_forward_validation import walk_forward_val

from visualization import plot_all_results

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import logging
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    

logging.getLogger("cmdstanpy").disabled = True

if __name__ == '__main__':
    
    with open('config.yaml', 'r') as f:
        paths = yaml.safe_load(f)
        
    df = pd.read_csv(paths['case_study_1']['raw_data'])
    
    df_cleaned = clean_data(df)
    
    train, test = preprocess_data(df_cleaned)
    
    X_train = train.drop(columns=['Close'])
    y_train = train['Close']
    
    X_test = test.drop(columns=['Close'])
    y_test = test['Close']
        
    models = [
        ("Random Forest", RandomForestRegressor(criterion='friedman_mse', max_depth=10, n_estimators=200)),
        ("SVR", SVR(C=10, kernel='linear', gamma='auto')),
        ("XGBoost", XGBRegressor(n_estimators = 300, learning_rate = 0.05, early_stopping_rounds = 50))
    ]
    
    ################ 1 - Train & Test Split #####################
    
    print("\n########### Train & Test Split Results ###########\n")
    
    y_preds = []
    
    for model_name, model in models:
        if model_name == "XGBoost":
            model.fit(X_train.drop(columns=['Date']), y_train, eval_set = [(X_test.drop(columns=['Date']), y_test)], verbose = False)
        else:
            model.fit(X_train.drop(columns=['Date']), y_train)
            
        y_pred = model.predict(X_test.drop(columns=['Date']))
        y_preds.append(y_pred)
        evaluate_model(model_name, test['Date'], y_test, y_pred, plot = False)
        print()

    y_pred = arimax_train_test(X_train, X_test, y_train, y_test, order=(2, 1, 0))
    y_preds.append(y_pred)
    evaluate_model('ARIMA', test['Date'], y_test, y_pred, plot = False)
    print()
    
    y_pred = prophet_train_test(train, test)
    y_preds.append(y_pred)
    evaluate_model('Prophet', test['Date'], y_test, y_pred, plot = False)
    
    # Plot all results
    plot_all_results(['Random Forest', 'SVR', 'XGBoost', 'ARIMA', 'Prophet'], test['Date'], y_test, y_preds, 'Model Comparison (Train & Test Split)', paths['case_study_1']['graphs']['results']['train_test_split'])
    
    # ################ 2 - Cross Validation   #####################
    
    tss = TimeSeriesSplit(n_splits= 5, gap=0)
    
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
        
    print("\n########### Cross Validation Results ###########\n")
            
    for model_name, model in models:
        if model_name == "XGBoost":
            model = XGBRegressor(n_estimators = 300, learning_rate = 0.05)
            preds, y_tests = cross_val_time_series(model, X.drop(columns = ['Date']), y, tss)
        else:
            preds, y_tests = cross_val_time_series(model, X.drop(columns = ['Date']), y, tss)
        print(f"\n{model_name}:")
        evaluate_cross_val(preds, y_tests)
    
    print("\nARIMAX:")
    preds_arimax, y_tests_arimax = arimax_cross_val(X.drop(columns = ['Date']), y, tss, order=(2, 1, 0))
    evaluate_cross_val(preds_arimax, y_tests_arimax)
    
    print("\nProphet:")
    preds_prophet, y_tests_prophet = prophet_cross_val(X, y, tss)
    evaluate_cross_val(preds_prophet, y_tests_prophet)
    
    ################ 3 - Walk Forward Validation ################

    print("\n########### Walk Forward Validation Results ###########\n")
    
    y_preds = []

    for model_name, model in models:
        if model_name == "XGBoost":
            model = XGBRegressor(n_estimators = 300, learning_rate = 0.05)
        
        y_pred = walk_forward_val(model, X_train.drop(columns=['Date']), X_test.drop(columns=['Date']), y_train, y_test)
        y_preds.append(y_pred)
        evaluate_model(model_name, test['Date'], y_test, y_pred, plot=False)
        print()
    
    y_pred = arima_walk_forward_val(y_train, y_test, order = (2, 1, 0))
    y_preds.append(y_pred)
    evaluate_model("ARIMA", test['Date'], y_test, y_pred, plot = False)
    print()
    
    y_pred = prophet_walk_forward_val(train, test)
    y_preds.append(y_pred)
    evaluate_model("Prophet", test['Date'], y_test, y_pred, plot = False)
    print()
    
    plot_all_results(['Random Forest', 'SVR', 'XGBoost', 'ARIMA', 'Prophet'], test['Date'], y_test, y_preds, 'Model Comparison (Walk Forward Validation)', paths['case_study_1']['graphs']['results']['walk_forward_val'])