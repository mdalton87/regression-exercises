import pandas as pd
import numpy as np
import os
import explore as ex

# acquire
from env import host, user, password
from pydataset import data

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Aquire Data

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'    
       

# Zillow Database        
        
def new_zillow_data():
    '''
    This function reads in the teclo_churn data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = '''
                SELECT *
                FROM  properties_2017
                JOIN predictions_2017 USING(parcelid)
                WHERE month(transactiondate) >= 05 and month(transactiondate) <= 06
                ;
                '''
    
    return pd.read_sql(sql_query, get_connection('zillow'))
        
        
        
def get_zillow_data(cached=False):
    '''
    This function reads in zillow data from Codeup database and writes data to a csv file if cached == False or if cached == True reads in titanic df from a csv file, returns df.
    '''
    if cached == False or os.path.isfile('zillow.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    return df


# Prepare Data





def wrangle_zillow():
    '''
    This functions creates a dataframe from the zillow dataset in the Codeup SQL database and preps the data for exploration. After retrieving the data, columns that contain > 15% null-values are dropped, then the dataframe is limited to the desired features, the parcelid is set to the index, columns names are renamed for clarity, rows with null-values are then dropped due to the low number compared to the dataset, the fips, zip code and year built feaures are converted to integers, and added an age of home feature that takes the year_built from the current year. Then outliers from the square_feet and tax_value are removed.
                 
    '''
    df = get_zillow_data(cached=True)
    df = df.dropna(axis=1,thresh=18653)
    features = ['parcelid', 'bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'fips', 'lotsizesquarefeet','latitude', 'longitude', 'regionidzip', 'yearbuilt', 'taxvaluedollarcnt', 'transactiondate']
    df = df[features]
    df.set_index('parcelid', inplace=True)
    df.columns = ['bathrooms', 'bedrooms', 'square_feet', 'fips', 'lot_size_sqft', 'latitude', 'longitude', 'zip_code', 'year_built', 'tax_value', 'transaction_date']
    df = df.dropna()
    df.fips = df.fips.astype(int)
    df.zip_code = df.zip_code.astype(int)
    df.year_built = df.year_built.astype(int)
    df['age_of_home'] = (2021 - df.year_built)
    df = ex.remove_outliers(df, 'square_feet', multiplier=1.5)
    df = ex.remove_outliers(df, 'tax_value', multiplier=1.5)
    
    return df


# Split Data

def impute_mode():
   '''
   impute mode for tax_value
   '''
   imputer = SimpleImputer(strategy='most_frequent')
   train[['tax_value']] = imputer.fit_transform(train[['tax_value']])
   validate[['tax_value']] = imputer.transform(validate[['tax_value']])
   test[['tax_value']] = imputer.transform(test[['tax_value']])
   return train, validate, test



def train_validate_test_split(df, seed=123):
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed)
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
    )
    return train, validate, test


def prep_zillow_data():
    df = wrangle_zillow()
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    train, validate, test = impute_mode()
    return train, validate, test
