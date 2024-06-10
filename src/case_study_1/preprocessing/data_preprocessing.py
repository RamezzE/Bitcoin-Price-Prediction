import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, SelectKBest

def scale_features(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled

def encode_categorical_features(df):
    encoder = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = encoder.fit_transform(df[col])
    
    return df
    
def extract_date_features(df):
        
    df['Date'] = pd.to_datetime(df['Date'])
    
    df_date = df['Date']
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['IsWeekend'] = df['Date'].dt.weekday >= 5
    df['IsStartOfMonth'] = df['Date'].dt.is_month_start
    df['IsEndOfMonth'] = df['Date'].dt.is_month_end

    df.drop(columns=['Date'], inplace=True)
    
    return df, df_date

def select_features(X_train, X_test, y_train, k = 10):

    selector = SelectKBest(score_func=mutual_info_regression, k=k)

    X_train_selectK = pd.DataFrame(selector.fit_transform(X_train, y_train), columns=X_train.columns[selector.get_support()])
    X_test_selectK = pd.DataFrame(selector.transform(X_test), columns=X_test.columns[selector.get_support()])

    return X_train_selectK, X_test_selectK

def preprocess_data(df):
    '''
    This function preprocesses the data by:
    - Extracting date features
    - Encoding categorical variables
    - Scaling features
    - Selecting K best features
    - Train, test split
    Returns the preprocessed train and test data.
    '''
    
    # Sorting the data by date
    df = df.iloc[::-1]

    # Handling missing values
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    df['Volume'] = df['Volume'].interpolate(method='linear', limit_direction='both')
    
    # Extracting date features
    
    df, df_date = extract_date_features(df)
        
    # Encoding categorical variables
    df = encode_categorical_features(df)
    
    # Train, test split 
    X = df.drop(columns=['Close'])
    y = df['Close']
    
    X = pd.concat([df_date, X], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle = False)

    train_dates = X_train['Date']
    test_dates = X_test['Date']
    
    X_train.drop(columns=['Date'], inplace=True)
    X_test.drop(columns=['Date'], inplace=True)

    # Scaling features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    X_train_selected, X_test_selected = select_features(X_train_scaled, X_test_scaled, y_train, k = 15)
    
    train_dates.reset_index(drop=True, inplace=True)
    test_dates.reset_index(drop=True, inplace=True)
    X_train_selected.reset_index(drop=True, inplace=True)
    X_test_selected.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    
    train = pd.concat([train_dates, X_train_selected, y_train], axis=1)
    test = pd.concat([test_dates, X_test_selected, y_test], axis=1)
    
    train.dropna(inplace=True)
    
    return train, test


if __name__ == '__main__':
    
    with open('config.yaml', 'r') as file:
        paths = yaml.safe_load(file)
    
    df = pd.read_csv(paths['case_study_1']['cleaned_data'])
    
    train, test = preprocess_data(df)    
    
    train.to_csv(paths['case_study_1']['processed-train-data'], index=False)
    test.to_csv(paths['case_study_1']['processed-test-data'], index=False)