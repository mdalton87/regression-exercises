# Zillow® Project - readme.md
![](http://zillow.mediaroom.com/image/Zillow_Wordmark_Blue_RGB.jpg)

***

## Project Description:
- The purpose of this project is to build a machine learning model that predicts the value of single unit properties that the tax district assesses using the property data from properties sold and purchased from May-August, 2017.

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
## Data Dictionary

### Data for Predicting Tax Value of Property
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| parcelid | Unique identifier for parcels (lots) | Index/int | 
| bathroomcnt | Number of bathrooms in home including fractional bathrooms | float |
| bedroomcnt | Number of bedrooms in home | float |
| square_feet | Calculated total finished living area of the home | float |
| fips | Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details | int |
| latitude | Latitude of the middle of the parcel multiplied by 10e6 | float |
| longitude | Longitude of the middle of the parcel multiplied by 10e6 | float |
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
## Data 

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



****
***
****

### Univariate:
Established variables for quant_vars, cat_vars and target for future exploratory use.
```json
{
quant_vars = ['tenure','monthly_charges','total_charges']
cat_vars = list((df.columns).drop(quant_vars))
target = 'churn'
}
```
```json
{
explore.explore_univariate(train, cat_vars, quant_vars)
}
```
#### Observations:
- There are significantly more non-senior citizens than senior citizens
- There are a lot more customers with dependents
- Significantly more customers with phone service than without
- Less have online security, online backup, device protection, and tech support
- A lot more people churn than stay
- More customers are Month-to-month than in contracts
- Electronis check is the most popular payment method

#### Questions:
- Customers with phone service that have multiple lines?
- Customers with internet that have online services (i.e. online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies)

### Bivariate:
Initially ran the explore_bivariate function and noticed a pattern. I decided to act limit the categorical variables and re-run the function.
```json
{
cat_vars = ['online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies']
}
```
```json
{
explore.explore_bivariate(train, target, cat_vars, quant_vars)
}
```
#### Observations:
- Variables with very low p-values (when scientific notation is used)
    - senior, partner, dependents, online_security, tech_support, paperless_billing, ***month-to_month***, fiber_optic_internet, one_year, two_year
- Low p-values (well below 0.05 but able to read without scientific notation)
    - online_backup, device_protection, streaming_tv, streaming_movies, 
- Barely passes 95% confidence (very close to alpha - 0.05)
    - multiple lines
- Does not pass
    - gender, phone_service
- Vast majority of churn happens before 30 months
- higher monthly bill increases churn

#### Questions:
- Do people with all online services churn more than customers without all of the online services?
- Are the really low p-values a good starting point?

***
## Question:

### Does the amount of online services affect churn rates of our customers with internet service?
#### Online services are:
   - online security
   - online backup
   - device protection
   - tech supprt
   - streaming tv
   - streaming movies

***   
## Answering the question:
### Cleaning train, validate and test. 
- Only need customers with internet service 
```json
{
train = train[train.internet_service_type != 'None']
validate = validate[validate.internet_service_type != 'None']
test = test[test.internet_service_type != 'None']
}
```
- Need to remove unecessary information for statistics and modeling
```json
{
dropcols = ['internet_service_type','senior_citizen','partner','dependents','tenure','phone_service','multiple_lines','paperless_billing','monthly_charges','total_charges','contract_type','payment_type','gender_male','one_year_contract','two_year_contract','credit_card_payment','e_check_payment','mailed_check_payment']
train = train.drop(columns=dropcols)
validate = validate.drop(columns=dropcols)
test = test.drop(columns=dropcols)
}
```
- Set the index to 'customer_id'
```json
{
train.set_index('customer_id')
validate.set_index('customer_id')
test.set_index('customer_id')
}
```
- Add a column that adds the number of online services that our internet customers have.
```json
{
train = train.assign(n_services = train[train.columns[1:7]].sum(axis=1))
validate = validate.assign(n_services = validate[validate.columns[1:7]].sum(axis=1))
test = test.assign(n_services = test[test.columns[1:7]].sum(axis=1))
}
```

***
## Visualize the question:
Below is the code I used to create better visualizations than the explore_bivariate function could produce.
```json
{
features = ['online_security', 'online_backup', 'device_protection']
_, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), sharey=True)
for i, feature in enumerate(features):
    sns.barplot(feature, 'churn', data=train, ax=ax[i])
    ax[i].set_xlabel('')
    ax[i].set_ylabel('Churn Rate')
    ax[i].set_title(feature)
    ax[i].axhline(train.churn.mean(), ls='--', color='grey')
}
```
```json
{
features = ['tech_support', 'streaming_tv', 'streaming_movies']
_, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), sharey=True)
for i, feature in enumerate(features):
    sns.barplot(feature, 'churn', data=train, ax=ax[i])
    ax[i].set_xlabel('')
    ax[i].set_ylabel('Churn Rate')
    ax[i].set_title(feature)
    ax[i].axhline(train.churn.mean(), ls='--', color='grey')
}
```
```json
{
feature = 'n_services'
target = 'churn'
sns.barplot(x=feature, y=target, data=train)
plt.xlabel('Number of Online Services per Customer with Internet')
plt.ylabel('Churn Rate')
plt.title("Relationship of Churn to Number of Online Services")
plt.show()
}
```

***
## Statistical Analysis

### χ<sup>2</sup> test
 - Testing for independence between 2 categorical values.
 - Churn is categorical (i.e. can be 1 or 0)
 - n_services is categorical (i.e. can be integers from 0-6)

- #### Hypothesis:
    - The **null hypothesis** = Churn is independent of the number of online services per internet customer.
    - the **alternate hypothesis** = We assume that there is an association between churn and the number of online services.

- #### Confidence level and alpha value:
    - I established a 95% confidence level
    - alpha = 1 - confidence, therefore alpha is 0.05

- #### Results:
    - χ<sup>2</sup> = 242.75
    - p-value = 1.45 x 10<sup>-49</sup>
    - degrees of freedom = 6


We reject the null hypothesis and move forward with the alternative hypothesis: We assume that there is an association between churn and the number of online services

### T-test

- I am running a T-test to verify that there is a difference between having no additional online services vs. having any extra online service

- #### Hypothesis:
    - The **null hypothesis** = There is no difference between in the means of customers without any additional online services and  customers with any number of online services.
    - the **alternate hypothesis** = There is a difference in the means of customers with online services and those without online services.

- #### Confidence level and alpha value:
    - I established a 95% confidence level
    - alpha = 1 - confidence, therefore alpha is 0.05

- #### Results:
    - t-score = -2.17
    - p-value = 0.044

***
## Modeling:

### Baseline
- Began by importing my machine learning models
```json
{
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
}
```
- Then I established a baseline:
    - First is to run a value_counts() on the column 'churn'
    - Then add a most_frequent column to the train dataframe and set all values set to the highest value_count(). (in this case there were 2122 - 0's and 983 - 1's)
    - Finally run the baseline_accuracy:
```json
{
train["most_frequent"] = 0
baseline_accuracy = (train.churn == train.most_frequent).mean()
print(f'My baseline prediction is survived = 0')
print(f'My baseline accuracy is: {baseline_accuracy:.2%}')
}
```
    - The goal here is to create a model that beats the baseline:
        - My baseline accuracy is: 68.34%
        
### Make: X_train, X_validate, X_test, y_train, y_validate, and y_test
```json 
{
X_train = train.drop(columns=['churn','most_frequent','customer_id'])
y_train = train.churn

X_validate = validate.drop(columns=['churn','customer_id'])
y_validate = validate.churn

X_test = test.drop(columns=['churn','customer_id'])
y_test = test.churn
}
```

### kNN:
```json
{
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
}
```
- #### Results:
    - Accuracy of KNN classifier on training set n_neighbors set to 5: 0.71
    - Accuracy of KNN classifier on validate set with n_neighbors set to 5: 0.70

### Random Forest:
```json
{
rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=5,
                            max_depth=50, 
                            random_state=42)
}
```
- #### Results
    - Accuracy of random forest classifier on training set: 0.73
    - Accuracy of random forest classifier on the validate set: 0.72

### Decision Tree:
```json
{
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
}
```
- #### Results
    - Accuracy of Decision Tree classifier on training set: 0.73
    - Accuracy of Decision Tree classifier on validate set: 0.72

### Logistic Regression:
```json
{
logit = LogisticRegression(penalty='l2', C=1, random_state=42, solver='lbfgs')    
}
```
- #### Results
    - Accuracy of on training set: 0.72
    - Accuracy out-of-sample set: 0.73
    - Accuracy of on test set: 0.72
    
***    
## Predictions CSV
Make a new dataframe
```json
{
    new_df = df
}
```
Limit the features to the features being tested
```json 
{
features = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'dsl_internet', 'fiber_optic_internet', 'n_services']
new_df_trimmed = new_df[new_df.internet_service_type != 'None']
new_df_trimmed = new_df_trimmed.drop(columns=dropcols)
new_df_trimmed = new_df_trimmed.assign(n_services = new_df_trimmed[new_df_trimmed.columns[1:7]].sum(axis=1))
}
```
Remove customers without internet service into prediction dataframe
```json
{
prediction_df = new_df[new_df.internet_service_type != "None"] 
}
```
Run predictions on the Logistic Regression model
```json
{
prediction_df['prediction'] = logit.predict(new_df_trimmed[features])   
}
``` 

```json
{
predictions = prediction_df[['customer_id','prediction']]
predictions.to_csv('predictions.csv')
}
```

***
## Key Takaways

- It is clear that fewer people churn when they have more online services. 

- What can we do now?

    - Promote online services into bundled packages:
        - For instance: the "Security package" will contain: online_security, online_backup, device_protection and tech_support
        - Along with "Streaming package" that will contain: streaming_tv, streaming_movies, and tech_support as well.

- With additional time dedicated to this project:

    - Investigate fiber optic customers in greater detail and look at possible combinations of factors that might be driving churn within that group.
    - Investigate our pricing structure of all internet service types and online services.
    - Make improvements to this report with more comments, markdown cells, and summary tables.
    - Make improvements to the cooresponding readme.md for this github containing project description with a more in depth explanation of how someone else can recreate this project and findings, and key takeaways from this project.
    - Add models to test varying hyperparameters and features to improve model performance.