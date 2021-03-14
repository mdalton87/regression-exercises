import pandas as pd
import numpy as np
import os

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





def wrangle_zillow(filter=1):
    '''

                 
    '''



# Split Data

def impute_mode():
   '''
   impute mode for taxvaluedollarcnt
   '''
   imputer = SimpleImputer(strategy='most_frequent')
   train[['taxvaluedollarcnt']] = imputer.fit_transform(train[['taxvaluedollarcnt']])
   validate[['taxvaluedollarcnt']] = imputer.transform(validate[['taxvaluedollarcnt']])
   test[['taxvaluedollarcnt']] = imputer.transform(test[['taxvaluedollarcnt']])
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
