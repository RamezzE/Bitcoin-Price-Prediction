import pandas as pd
import yaml

def clean_data(df):
    
    '''
    This function cleans the raw data by removing 
    duplicates and replacing missing values.
    Returns the cleaned dataframe.
    '''
    
    df.drop_duplicates(inplace=True)
    
    df['Volume'] = df['Volume'].str.replace(',', '')
    df['Market Cap'] = df['Market Cap'].str.replace(',', '')
    
    df.loc[df['Volume'] == '-', 'Volume'] = None
    
    return df

if __name__ == '__main__':
    
    with open('config.yaml', 'r') as f:
        paths = yaml.safe_load(f)
        
    df = pd.read_csv(paths['case_study_1']['raw_data'])

    df_cleaned = clean_data(df)

    df_cleaned.to_csv(paths['case_study_1']['cleaned_data'], index=False)
