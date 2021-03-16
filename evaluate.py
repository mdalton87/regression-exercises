import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt
import seaborn as sns
import warnings
from pydataset import data



def plot_residuals(df, model, x, y):
    '''
    Parameters: y = df['target_variable']
    '''
    df['baseline'] = y.mean()
    df['residual'] = df.yhat - y
    df['residual_baseline'] = df.baseline - y
    df['yhat'] = model.predict(x)
    
    plt.subplots(figsize=(10,8))
    
    plt.subplot(2,1,1)
    plt.scatter(x, df.residual)
    plt.axhline(y = 0, ls = ':')
    plt.title('OLS model residuals')
    
    plt.subplot(2,1,2)
    plt.scatter(x, df.residual_baseline)
    plt.axhline(y = 0, ls = ':')
    plt.title('Baseline Residuals')
    
    plt.show()
    
    
    
def plot_baseline_OLS_model(df, x, y):  
    '''
    Parameters: x = df['variable_being_predicted']
                y = df['target_variable']
    '''
    fig = plt.figure(figsize = (10,6))
    plt.scatter(x, y)
    plt.axhline(y = y.mean(), ls = ':')
    plt.plot(x, df.yhat)
    plt.text(8,21, 'baseline(mean)')
    plt.text(8.25,40, 'Regression model')
    plt.xlabel('x = exam1 score')
    plt.ylabel('y = final score')
    plt.title('Baseline and OLS regression model')
    
    
def regression_errors(df, y):
    '''
    Parameters: y = df['target_variable']
                yhat = model.predict(x), where x = df['variable_being_predicted']
    '''
    SSE = mean_squared_error(y, df['yhat'])*len(df)
    MSE = SSE/len(df)
    ESS = sum((df.yhat - y.mean())**2)
    TSS = ESS + SSE
    RMSE = sqrt(mean_squared_error(y, df.yhat))
    R2 = ESS/TSS
    print(f'The Sum of Squared Errors is {SSE},\nThe Mean Squared Error is {MSE},\nThe Explained Sum of Squares = {ESS},\nThe Total Sum of Squares is {TSS},\nThe Root Mean Squared Error is {RMSE},\nand The R-squared value is {R2}')
    return SSE, MSE, ESS, TSS, RMSE, R2

    
def baseline_mean_errors(df, y):
    '''
    Parameters: y = df['target_variable']
                yhat = model.predict(x), where x = df['variable_being_predicted']  
    '''
    SSE_baseline = mean_squared_error(y, df['baseline'])*len(df)
    MSE_baseline = SSE_baseline/len(df)
    RMSE_baseline = sqrt(mean_squared_error(y, df.baseline))
    print(f'The baseline for the Sum of Squared Errors is {SSE_baseline},\nThe baseline for the Mean Squared Error is {MSE_baseline},\nand the baseline for the Root Mean Squared Error is {RMSE_baseline}')
    return SSE_baseline, MSE_baseline, RMSE_baseline

        
def better_than_baseline(SSE, MSE, RMSE, SSE_baseline, MSE_baseline, RMSE_baseline):
    df_eval = pd.DataFrame(np.array(['SSE','MSE','RMSE']), columns=['metric'])
    df_baseline_eval = pd.DataFrame(np.array(['SSE_baseline','MSE_baseline','RMSE_baseline']), columns=['metric'])
    
    df_eval['model_error'] = np.array([SSE, MSE, RMSE])
    df_baseline_eval['model_error'] = np.array([SSE_baseline, MSE_baseline, RMSE_baseline])
    df_eval['error_delta'] = df_eval.model_error - df_baseline_eval.model_error
    df_eval['is_better_than_baseline'] = df_eval['error_delta'] < 0 
    return df_eval
    
    
def model_significance(model, df, y): 
    f_pval = model.f_pvalue
    evs = explained_variance_score(y, df.yhat)
    print(f"p-value for model significance is {f_pval}\n")
    print(f'The Explained Variance Score is {round(evs,3)}\n')
    print(model.summary())
    
    