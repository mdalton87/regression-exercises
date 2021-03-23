# <a name="top"></a>Zillow® Project - readme.md
![](http://zillow.mediaroom.com/image/Zillow_Wordmark_Blue_RGB.jpg)

***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire, Prep, and Exploration](#wrangle)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___

***
## <a name="project_description"></a>Project Description:
[[Back to top](#top)]
- The purpose of this project is to build a machine learning model that predicts the value of single unit properties that the tax district assesses using the property data from properties sold and purchased from May-August, 2017.
- 

## Goals

The goals of the project are to answer the questions and deliver the following:

- A Complete Jupyter Notebook Report showing the step-by-step process of collecting and wrangling data, exploring the possible drivers of the tax value, run a statistical analysis of various features and finally, model and test the chosen feaures using common regression techniques.
- a README.md file containing: 
    - project description with goals
    - a data dictionary
    - project planning (lay out your process through the data science pipeline)
    - explanation of how someone else can recreate your project and findings
    - key findings and takeaways from your project.
- A wrangle.py file that holds my functions to acquire and prepare your data.
- Clearly state your starting hypotheses (and add the testing of these to your task list).

***
## <a name="planning"></a>Project Planning: 
[[Back to top](#top)]

### Projet Outline:
- Acquisiton of data through Codeup SQL Server, using env.py file with username, password, and host
- Prepare and clean data with python - Jupyter Labs Notebook
- Explore data
    - if value are what the dictionary says they are
    - null values
        - are the fixable or should they just be deleted
    - categorical or continuous values
    - Make graphs that show 
- Run statistical analysis
- Model data 
- Test Data
- Conclude results
        
### Hypothesis
- I will get a R<sup>2</sup> > .40
- square feet will be the best driver for tax value
- total number of beds and baths will be better driver for tax value than either bedrooms or bathrooms alone. 

### Target variable
- tax_value 

### Need to haves:
- bedrooms
- bathrooms
- square_feet

### Nice to haves:
- latitude 
- longitude
- zip_code (regionidzip)
    - not used because the zip codes did not match to the region in the data
- A completed dataset with no null values

***
## <a name="findings"></a>Key Findings:
[[Back to top](#top)]
- My best model was the Polynomial Regression model with a power of 2, 


***
## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]

### Data for Predicting Tax Value of Property
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| parcelid | Unique identifier for parcels (lots) | Index/int | 
| bathroomcnt | Number of bathrooms in home including fractional bathrooms | float |
| bedroomcnt | Number of bedrooms in home | float |
| square_feet | Calculated total finished living area of the home | float |
| latitude | Latitude of the middle of the parcel multiplied by 10<sup>6</sup> | float |
| longitude | Longitude of the middle of the parcel multiplied by 10<sup>6</sup> | float |
| year_built | The Year the principal residence was built | int |
| tax_value* | The total tax assessed value of the parcel | float |
| age_of_home | year_built minus 2021 | int |
| beds_and_baths | The sum of all bedrooms and bathrooms | float |
| beds_per_sqft | The number of bedrooms divided by the square_feet | float |
| baths_per_sqft | The number of bathrooms divided by the square_feet | float |

\* - Indicates the target feature in this Zillow data.

### Data Dictionary for Calculating the Tax Rates of Each County

| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| fips | Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details | int |
| tax_value | The total tax assessed value of the parcel | float |
| tax_amount | The total property tax assessed for that assessment year | float |
| tax_rate | tax_amount / tax_value | float |


***
## <a name="wrangle"></a>Data Acquire, Preparation, and Exploration
[[Back to top](#top)]

### Acqisition and Preparation
- wrangle.py

| Function Name | Purpose |
| ----- | ----- |
| wrangle_zillow() | This functions creates a dataframe from the zillow dataset in the Codeup SQL database and preps the data for exploration. The SQL query filters records by date and single unit homes, features that contain > 15% null-values are immediately dropped, then the dataframe is limited to the desired features, the parcelid is set to the index, columns names are renamed for clarity, rows with null-values are then dropped due to the low number compared to the dataset, the fips, zip code and year built feaures are converted to integers, outliers are then removed from square_feet and tax_value, and finally I added new features: age_of_home, beds_and_baths, beds_per_sqft, baths_per_sqft |
| tax_rate_distribution() | This function creates the dataframe used to calculate the tax distribution rate per county. It takes in the cached zillow dataset, sets the parcelid as the index, makes the features list in order to limit the dataframe, renames the columns for clarity, drops null values, creates the tax_rate feature, and removed outliers from tax_rate and tx_value. |


***

## Data Exploration:
- wrangle.py 
| Function Name | Definition |
| ----- | ----- |
| select_kbest | This function takes in a dataframe, the target feature as a string, and an interger (k) that must be less than or equal to the number of features and returns the (k) best features |
| rfe | This function takes in a dataframe, the target feature as a string, and an interger (k) that must be less than or equal to the number of features and returns the best features by making a model, removing the weakest feature, then, making a new model, and removing the weakest feature, and so on. |
| train_validate_test_split | This function takes in a dataframe, the target feature as a string, and a seed interger and returns split data: train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test |
| get_object_cols() | This function takes in a dataframe and identifies the columns that are object types and returns a list of those column names. |
| get_numeric_cols(X_train, object_cols) | This function takes in a dataframe and list of object column names and returns a list of all other columns names, the non-objects. |
| min_max_scale(X_train, X_validate, X_test, numeric_cols) | This function takes in 3 dataframes with the same columns, a list of numeric column names (because the scaler can only work with numeric columns), and fits a min-max scaler to the first dataframe and transforms all 3 dataframes using that scaler. It returns 3 dataframes with the same column names and scaled values. 


### Select K Best Results:
- bathrooms
- bedrooms
- square_feet
- year_built
- age_of_home
- beds_and_baths
- beds_per_sqft

### RFE Results:
- bathrooms
- bedrooms
- square_feet
- year_built
- age_of_home
- beds_and_baths
- baths_per_sqft

### Object_cols:
- Yeilds an empty list
- Used in order to run numeric_cols function without issues

### Numeric_cols:
- bathrooms
- bedrooms
- square_feet
- latitude
- longitude
- year_built
- age_of_home
- beds_and_baths
- beds_per_sqft
- baths_per_sqft

***

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]

### Correlation Test
 - Used to check if two samples are related. They are often used for feature selection and multivariate analysis in data preprocessing and exploration.
 - This test returns the correlation coefficient (r) nd a p-value (p)
     - the correlation coefficient is used to measure how strong a relationship is between two variables
     - the p-value is the probability of obtaining test results at least as extreme as the results actually observed, under the assumption that the null hypothesis is correct
- I will run a correlation test for each of the following features against the target (tax_value):
    - bathrooms
    - bedrooms
    - square_feet
    - beds_and_baths
    - beds_per_sqft
    - baths_per_sqft

#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is that there is no correlation between the two samples.
- The alternate hypothesis (H<sub>1</sub>) is that there is a correlation between the two samples.


#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
 - The numbers:
 
 | Feature | r | p-value |
 | ---- | ---- | ---- |
 | bathrooms | 0.4431 | 0.0 |
 | bedrooms | 0.2590 | 0.0 |
 | square_feet | 0.5296 | 0.0 |
 | beds_and_baths | 0.3871 | 0.0 |
 | beds_per_sqft | -0.4896 | 0.0 |
 | baths_per_sqft | -0.3893 | 0.0 |

 - Summary:
     - All correlation tests reject the H<sub>0</sub> because all p-values were less then the alpha of 0.05. 
     - Based on the correlation coefficient, square feet of the property appears to be the best driver for increasing the tax value despite only having a medium positive correlation. 
     - A medium negative correlation with beds_per_sqft is evident with the r = -0.4896. 
     - Number of bathrooms is a better driver than number of bedrooms
     - Number of bedrooms has the weakest correlation with 0.259

### T-test
- A T-test allows me to compare a categorical and a continuous variable by comparing the mean of the continuous variable by subgroups based on the categorical variable
- The t-test returns the t-statistic and the p-value:
    - t-statistic: 
        - Is the ratio of the departure of the estimated value of a parameter from its hypothesized value to its standard error. It is used in hypothesis testing via Student's t-test. 
        - It is used in a t-test to determine if you should support or reject the null hypothesis
        - t-statistic of 0 = H<sub>0</sub>
    -  - the p-value:
        - The probability of obtaining test results at least as extreme as the results actually observed, under the assumption that the null hypothesis is correct
- I wanted to know if LA County's tax rate is greater than the tax rate of all 3 counties which by definition a 1 sample t-test: 
    - Ventura County
    - Orange County
    - LA County
- In order to run this test I would use the tax_rate_distribution() function to create a dataframe that include fips, tax amount, and tax value to calculate the tax rate and still have the county data (fips).


#### Hypothesis:
- The H<sub>0</sub> is that there is no difference in the means of the LA County Tax rates.
- The H<sub>1</sub> is that the LA County Tax Rate is a different mean than the entire population.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- The numvbers:
    - t-statistic = 46.5759
    - p-value = 0.0
    
- Summary;
    - There is enough evidence to reject the H<sub>0</sub>:
        - The t-statistic is far enough away from 0 (far enough from the H<sub>0</sub>)
        - The p-value is 0.0 which is less than 0.05 
    - I can move forward with the H<sub>1</sub> indicating that there is a difference in the tax_rates of LA County compared to all 3 counties.

***
## <a name="model"></a>Modeling:
[[Back to top](#top)]

Regression is supervised machine learning technique for predicting a continuous target variable. Since the target, tax_value, is a continuous variable regression model are the ideal choice for this project.



### Baseline


- Begin by creating columns with the predicted mean and median within the y_train and y_validate dataframes
    
```json
{
value_pred_mean = y_train.tax_value.mean()
y_train['value_pred_mean'] = value_pred_mean
y_validate['value_pred_mean'] = value_pred_mean

value_pred_median = y_train.tax_value.median()
y_train['value_pred_median'] = value_pred_median
y_validate['value_pred_median'] = value_pred_median
}
```
- Run a Root Mean Squared Error (RMSE) on both the mean and median

```json
{
rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_mean) ** (1/2)
rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_mean) ** (1/2)

rmse_train_baseline_mean = rmse_train
rmse_validate_baseline_mean = rmse_validate

print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_median) ** (1/2)
rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_median) ** (1/2)

rmse_train_baseline_median = rmse_train
rmse_validate_baseline_median = rmse_validate

print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
}
```

- Baseline: 
    - RMSE using Mean
        - Train/In-Sample:  **271194.48**
        - Validate/Out-of-Sample:  **272149.78**
    - RMSE using Median
        - Train/In-Sample:  **276269.48** 
        - Validate/Out-of-Sample:  **277446.89**
        
***

### Models and R<sup>2</sup> Values:
- Will run the following models:
    - LinearRegression (OLS)
    - TweedieRegressor (GLM)
    - LassoLars
    - Polynomial Regression
- R<sup>2</sup> Value is the coefficient of determination, pronounced "R squared", is the proportion of the variance in the dependent variable that is predictable from the independent variable. 
    - Essentially it is a statistical measure of how close the data are to the fitted regression line.
#### LinearRegression (OLS)

```json 
{
lm = LinearRegression(normalize=True)


lm.fit(X_train, y_train.tax_value)

y_train['value_pred_lm'] = lm.predict(X_train)

rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_lm) ** (1/2)

y_validate['value_pred_lm'] = lm.predict(X_validate)

rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_lm) ** (1/2)

print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", round(rmse_train, 4), 
      "\nValidation/Out-of-Sample: ", round(rmse_validate, 4))
}
```
- RMSE for OLS using LinearRegression:
    - Training/In-Sample:  **217503.9051**
    - Validation/Out-of-Sample:  **220468.9564**
- R<sup>2</sup> Value = **0.3437**


### TweedieRegressor (GLM):
```json
{
glm = TweedieRegressor(power=0, alpha=0)

glm.fit(X_train, y_train.tax_value)

y_train['value_pred_glm'] = glm.predict(X_train)

rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_glm) ** (1/2)

y_validate['value_pred_glm'] = glm.predict(X_validate)

rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_glm) ** (1/2)

print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", round(rmse_train,4), 
      "\nValidation/Out-of-Sample: ", round(rmse_validate, 4))
}
```
- RMSE for GLM using Tweedie, power=1 & alpha=0
    - Training/In-Sample:  **217516.6069**
    - Validation/Out-of-Sample:  **220563.6468**
- R<sup>2</sup> Value = **0.3432**

### LassoLars:
```json
{
lars = LassoLars(alpha=1.0)

lars.fit(X_train, y_train.tax_value)

y_train['value_pred_lars'] = lars.predict(X_train)

rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_lars) ** (1/2)

y_validate['value_pred_lars'] = lars.predict(X_validate)

rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_lars) ** (1/2)

print("RMSE for Lasso + Lars\nTraining/In-Sample: ", round(rmse_train, 4), 
      "\nValidation/Out-of-Sample: ", round(rmse_validate,4))
}
```
- RMSE for Lasso + Lars
    - Training/In-Sample:  **217521.8752**
    - Validation/Out-of-Sample:  **220536.3882**
- R<sup>2</sup> Value = **0.3433**

### Polynomial Regression:
```json
{
pf = PolynomialFeatures(degree=2)

X_train_degree2 = pf.fit_transform(X_train)

X_validate_degree2 = pf.transform(X_validate)
X_test_degree2 = pf.transform(X_test)


lm2 = LinearRegression(normalize=True)

lm2.fit(X_train_degree2, y_train.tax_value)

y_train['value_pred_lm2'] = lm2.predict(X_train_degree2)

rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_lm2) ** (1/2)

y_validate['value_pred_lm2'] = lm2.predict(X_validate_degree2)

rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_lm2) ** (1/2)

print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", round(rmse_train,4), 
      "\nValidation/Out-of-Sample: ", round(rmse_validate,4))
}
```
- RMSE for Polynomial Model, degrees=2
    - Training/In-Sample:  **211227.5585**
    - Validation/Out-of-Sample:  **214109.6968**
- R<sup>2</sup> Value = **0.3810**


## Selecting the Best Model:

| Model | Training/In Sample RMSE | Validation/Out of Sample RMSE | R<sup>2</sup> Value |
| ---- | ----| ---- | ---- |
| Baseline | 271194.48 | 272149.78 | -2.1456 x 10<sup>-5</sup> |
| Linear Regression | 217503.9051 | 220468.9564 | 0.3437 |
| Tweedie Regressor (GLM) | 217516.6069 | 220563.6468 | 0.3432 |
| Lasso Lars | 217521.8752 | 220536.3882 | 0.3433 |
| Polynomial Regression | 211227.5585 | 214109.6968 | 0.3810 |

- The best model will have the lowest error and the highest R<sup>2</sup>
- In this case the best model is the Polynomial Regression Model

## Testing the Model
```json
{
y_test = pd.DataFrame(y_test)

y_test['value_pred_lm2'] = lm2.predict(X_test_degree2)

rmse_test = mean_squared_error(y_test.tax_value, y_test.value_pred_lm2) ** (1/2)

print("RMSE for LassoLars Model\nOut-of-Sample Performance: ", rmse_test)    

}
```
- RMSE for LassoLars Model
     - Out-of-Sample Performance:  **213615.6212**


***

## <a name="conclusion"></a>Conclusion:
[[Back to top](#top)]

 