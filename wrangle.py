import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression, RFE, SelectKBest
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler 

from env import host, user, password

from math import sqrt
import seaborn as sns
import warnings
from pydataset import data
import os


# Aquire Data

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'    
       

# Telco Database        
        
def new_telco_data():
    '''
    This function reads in the teclo_churn data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = '''
    SELECT *
    FROM customers
    JOIN contract_types USING(`contract_type_id`)
    JOIN internet_service_types USING(`internet_service_type_id`)
    JOIN payment_types USING(payment_type_id);
                '''
    
    return pd.read_sql(sql_query, get_connection('telco_churn'))
        
        
        
def get_telco_data(cached=False):
    '''
    This function reads in telco_churn data from Codeup database and writes data to a csv file if cached == False or if cached == True reads in titanic df from a csv file, returns df.
    '''
    if cached == False or os.path.isfile('telco_churn.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_telco_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('telco_churn.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('telco_churn.csv', index_col=0)
        
    return df


# Prepare Data


def telco_two_yr_contract():
    '''
This function acquires telco_churn.csv, filters out all customers without a two year contract, converts 'total_charges' from object to float and replaces nulls with 0, since these are new customers and have yet to pay their first months charges, and returns a dataframe with the specified features.
    '''
    # acquires telco_churn.csv and calls it df
    df = get_telco_data(cached=True)
    
    # annotates features to be used
    features = ['customer_id', 'monthly_charges', 'tenure', 'total_charges']
    
    # Filter out all customers without a two year contract:
        # contract_type_id == 3 is customers with Two year contracts
    df = df[df.contract_type_id == 3]
    
    # make df with only features annotated above
    df = df[features]
    
    # converts 'total_charges' from object to float and replaces nulls with 0, since these are new customers and have yet to pay their first months charges.
    df.total_charges = pd.to_numeric(df.total_charges, errors='coerce').astype('float64')
    df.total_charges = df.total_charges.fillna(value=0)
    return df


def telco_one_yr_contract():
    '''
This function acquires telco_churn.csv, filters out all customers without a one year contract, converts 'total_charges' from object to float and replaces nulls with 0, since these are new customers and have yet to pay their first months charges, and returns a dataframe with the specified features.
    '''
    # acquires telco_churn.csv and calls it df
    df = get_telco_data(cached=True)
    
    # annotates features to be used
    features = ['customer_id', 'monthly_charges', 'tenure', 'total_charges']
    
    # Filter out all customers without a one year contract:
        # contract_type_id == 2 is customers with one year contracts
    df = df[df.contract_type_id == 2]
    
    # make df with only features annotated above
    df = df[features]
    
    # converts 'total_charges' from object to float and replaces nulls with 0, since these are new customers and have yet to pay their first months charges.
    df.total_charges = pd.to_numeric(df.total_charges, errors='coerce').astype('float64')
    df.total_charges = df.total_charges.fillna(value=0)
    return df

def telco_month_contract():
    '''
This function acquires telco_churn.csv, filters out all customers without a month-to-month contract, converts 'total_charges' from object to float and replaces nulls with 0, since these are new customers and have yet to pay their first months charges, and returns a dataframe with the specified features.
    '''
    # acquires telco_churn.csv and calls it df
    df = get_telco_data(cached=True)
    
    # annotates features to be used
    features = ['customer_id', 'monthly_charges', 'tenure', 'total_charges']
    
    # Filter out all customers without a month-to-month contract:
        # contract_type_id == 1 is customers with month-to-month contracts
    df = df[df.contract_type_id == 1]
    
    # make df with only features annotated above
    df = df[features]
    
    # converts 'total_charges' from object to float and replaces nulls with 0, since these are new customers and have yet to pay their first months charges.
    df.total_charges = pd.to_numeric(df.total_charges, errors='coerce').astype('float64')
    df.total_charges = df.total_charges.fillna(value=0)
    return df

def telco_no_contract_filter():
    '''
This function acquires telco_churn.csv, converts 'total_charges' from object to float and replaces nulls with 0, since these are new customers and have yet to pay their first months charges, and returns a dataframe with the specified features.
    '''
    # acquires telco_churn.csv and calls it df
    df = get_telco_data(cached=True)
    
    # annotates features to be used
    features = ['customer_id', 'monthly_charges', 'tenure', 'total_charges']
    
    # make df with only features annotated above
    df = df[features]
    
    # converts 'total_charges' from object to float and replaces nulls with 0, since these are new customers and have yet to pay their first months charges.
    df.total_charges = pd.to_numeric(df.total_charges, errors='coerce').astype('float64')
    df.total_charges = df.total_charges.fillna(value=0)
    return df


def wrangle_telco(filter=1):
    '''
    This function takes in a filter value, reads the telco_churn data and returns a cleaned, filtered dataframe of the dataset \n
    Parameters:\n
        filter=: 0, Does not filter by contract type 
                 1, Removes all customers that do not have a Month-to-Month Contract
                 2, Removes all customers that do not have a One Year Contract
                 3, Removes all customers that do not have a Two Year Contract
                 
    '''
    if filter == 0:
        df = telco_no_contract_filter()
    elif filter == 1:
        df = telco_month_contract()
    elif filter == 2:
        df = telco_one_yr_contract()
    elif filter == 3:
        df = telco_two_yr_contract()
    else:
        print('To filter for Month-to-Month contracts: filter=1')
        print('To filter for One year contracts: filter=2')
        print('To filter for Two year contracts: filter=3')         
    return df


# Split Data

# def impute_mode():
#     '''
#     impute mode for churn
#     '''
#     imputer = SimpleImputer(strategy='most_frequent')
#     train[['total_charges']] = imputer.fit_transform(train[['total_charges']])
#     validate[['total_charges']] = imputer.transform(validate[['total_charges']])
#     test[['total_charges']] = imputer.transform(test[['total_charges']])
#     return train, validate, test



def train_validate_test_split(df, seed=42):
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed)
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
    )
    return train, validate, test


def prep_telco_data(x):
    df = wrangle_telco(filter=x)
    train_validate, test = train_test_split(df, test_size=.2, random_state=42)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=42)
#     train, validate, test = impute_mode()
    return train, validate, test




# Ryans wrangle for feature engineering

