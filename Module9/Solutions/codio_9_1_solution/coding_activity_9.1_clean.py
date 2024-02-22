# %% md
# ### Codio Activity 9.1: Sequential Feature Selection 
# 
# **Expected Time: 60 minutes**
# 
# **Total Points: 70**
# 
# This activity focuses on using the stepwise feature selection approach outlined in video 9.3.  Here, your goal is to first generate polynomial features for a `degree = 3` model and limit them to a set number using the `SequentialFeatureSelection`. For more information on the selector see [here](https://scikit-learn.org/stable/modules/feature_selection.html#sequential-feature-selection). 
# %% md
# #### Index 
# 
# - [Problem 1](#Problem-1)
# - [Problem 2](#Problem-2)
# - [Problem 3](#Problem-3)
# - [Problem 4](#Problem-4)
# - [Problem 5](#Problem-5)
# %%
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn import set_config

set_config(display="diagram")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %% md
# ### The Data
# 
# The dataset used is from kaggle [here](https://www.kaggle.com/mirichoi0218/insurance) and contains information on insurance clients and their medical costs billed by the insurance company.  You will build regression models to predict the logarithm of the `charges`.    The dataset is loaded and displayed below, and the original target is plotted next to its resulting logarithm.   
# %%
insurance = pd.read_csv('data/insurance.csv')
# %%
insurance.head()
# %%
insurance.info()
# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
ax[0].hist(insurance['charges'])
ax[0].grid()
ax[0].set_title('Original charges column')
ax[1].hist(np.log1p(insurance['charges']))
ax[1].grid()
ax[1].set_title('Logarithm of charges');
# %% md
# [Back to top](#Index)
# 
# ### Problem 1
# 
# #### Train/Test split
# 
# **Points: 10**
# 
# Create a train and test set using `train_test_split` and assign to `X_train`, `X_test`, `y_train`, `y_test` below with parameters:
# 
# - `test_size = 0.3`
# - `random_state = 42`
# 
# The target data should be transformed according to the `np.log1p` function.  
# %%
### GRADED

X_train, X_test, y_train, y_test = '', '', '', ''

# YOUR CODE HERE
X_train, X_test, y_train, y_test = train_test_split(insurance.drop('charges', axis=1), np.log1p(insurance.charges),
                                                    random_state=42, test_size=0.3)

# Answer check
print(X_train.shape)
print(X_test.shape)
# %%

# %% md
# [Back to top](#Index)
# 
# ### Problem 2
# 
# #### Building `PolynomialFeatures`
# 
# **Points: 10**
# 
# To begin, create a `DataFrame` with the `degree = 3` features and interaction terms added for numeric columns. Assign the following objects to the variables:
# 
# - `poly_features`: Instantiate a `PolynomialFeatures` object for quadratic features without a bias term.
# - `X_train_poly`: Fit and transformed features using `['age', 'bmi', 'children']`.  
# - `X_test_poly`: Transformed test data using `['age', 'bmi', 'children']`.
# - `poly_columns`: Column names of transformed data
# - `train_df`: DataFrame with training features and column names.
# - `test_df`: DataFrame with test features and column names
# 
# The DataFrame `train_df` should look like:
# %% md
# '<table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>age</th>      <th>bmi</th>      <th>children</th>      <th>age^2</th>      <th>age bmi</th>      <th>age children</th>      <th>bmi^2</th>      <th>bmi children</th>      <th>children^2</th>      <th>age^3</th>      <th>age^2 bmi</th>      <th>age^2 children</th>      <th>age bmi^2</th>      <th>age bmi children</th>      <th>age children^2</th>      <th>bmi^3</th>      <th>bmi^2 children</th>      <th>bmi children^2</th>      <th>children^3</th>    </tr>  </thead>  <tbody>    <tr>      <th>0</th>      <td>61.0</td>      <td>31.160</td>      <td>0.0</td>      <td>3721.0</td>      <td>1900.760</td>      <td>0.0</td>      <td>970.945600</td>      <td>0.00</td>      <td>0.0</td>      <td>226981.0</td>      <td>115946.360</td>      <td>0.0</td>      <td>59227.681600</td>      <td>0.0</td>      <td>0.0</td>      <td>30254.664896</td>      <td>0.0000</td>      <td>0.00</td>      <td>0.0</td>    </tr>    <tr>      <th>1</th>      <td>46.0</td>      <td>27.600</td>      <td>0.0</td>      <td>2116.0</td>      <td>1269.600</td>      <td>0.0</td>      <td>761.760000</td>      <td>0.00</td>      <td>0.0</td>      <td>97336.0</td>      <td>58401.600</td>      <td>0.0</td>      <td>35040.960000</td>      <td>0.0</td>      <td>0.0</td>      <td>21024.576000</td>      <td>0.0000</td>      <td>0.00</td>      <td>0.0</td>    </tr>    <tr>      <th>2</th>      <td>54.0</td>      <td>31.900</td>      <td>3.0</td>      <td>2916.0</td>      <td>1722.600</td>      <td>162.0</td>      <td>1017.610000</td>      <td>95.70</td>      <td>9.0</td>      <td>157464.0</td>      <td>93020.400</td>      <td>8748.0</td>      <td>54950.940000</td>      <td>5167.8</td>      <td>486.0</td>      <td>32461.759000</td>      <td>3052.8300</td>      <td>287.10</td>      <td>27.0</td>    </tr>    <tr>      <th>3</th>      <td>55.0</td>      <td>30.685</td>      <td>0.0</td>      <td>3025.0</td>      <td>1687.675</td>      <td>0.0</td>      <td>941.569225</td>      <td>0.00</td>      <td>0.0</td>      <td>166375.0</td>      <td>92822.125</td>      <td>0.0</td>      <td>51786.307375</td>      <td>0.0</td>      <td>0.0</td>      <td>28892.051669</td>      <td>0.0000</td>      <td>0.00</td>      <td>0.0</td>    </tr>    <tr>      <th>4</th>      <td>25.0</td>      <td>45.540</td>      <td>2.0</td>      <td>625.0</td>      <td>1138.500</td>      <td>50.0</td>      <td>2073.891600</td>      <td>91.08</td>      <td>4.0</td>      <td>15625.0</td>      <td>28462.500</td>      <td>1250.0</td>      <td>51847.290000</td>      <td>2277.0</td>      <td>100.0</td>      <td>94445.023464</td>      <td>4147.7832</td>      <td>182.16</td>      <td>8.0</td>    </tr>  </tbody></table>'
# %%
### GRADED

poly_features = ''
X_train_poly = ''
X_test_poly = ''
columns = ''
train_df = ''
test_df = ''

# YOUR CODE HERE
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train[['age', 'bmi', 'children']])
X_test_poly = poly_features.fit_transform(X_test[['age', 'bmi', 'children']])
columns = poly_features.get_feature_names_out()
train_df = pd.DataFrame(X_train_poly, columns=columns)
test_df = pd.DataFrame(X_test_poly, columns=columns)

test_df.to_csv('data/test_cubic.csv', index=False)
train_df.to_csv('data/train_cubic.csv', index=False)
# ANSWER CHECK
train_df.head()
# %%
test_df
# %%

# %% md
# [Back to top](#Index)
# 
# ### Problem 3
# 
# #### Using `SequentialFeatureSelector`
# 
# **Points: 10**
# 
# Now, using the polynomial features you will use forward feature selection to select three features (`n_features_to_select = 3`) using a `LinearRegression` estimator to perform the feature selection on the training data.  Assign your transformed features to `best_features_df` as a DataFrame with appropriate feature names.  The transformer should be instantiated as `selector` and the transformed training data should be assigned to `best_features` as an array.
# %%
### GRADED

selector = ''
best_features = ''
best_features_df = ''

# YOUR CODE HERE
selector = SequentialFeatureSelector(LinearRegression(), n_features_to_select=3)
best_features = selector.fit_transform(train_df, y_train)
best_features_df = pd.DataFrame(best_features, columns=selector.get_feature_names_out())

# ANSWER CHECK
best_features_df.head()
# %%

# %% md
# [Back to top](#Index)
# 
# ### Problem 4
# 
# #### `Pipeline` for regression model
# 
# **Points: 20**
# 
# 
# Now, create a `Pipeline` object with steps `column_selector` to select three features, and `linreg` to build a `LinearRegression` estimator.  Determine the Mean Squared Error on the train and test set respectively and assign to `train_mse` and `test_mse` as a float. Be sure to use `train_df` and `test_df` from above for fitting and predicting.
# %%
### GRADED

pipe = ''
train_preds = ''
test_preds = ''
train_mse = ''
test_mse = ''

# YOUR CODE HERE
pipe = Pipeline([('column_selector', selector), ('linreg', LinearRegression())])
pipe.fit(train_df, y_train)
train_preds = pipe.predict(train_df)
test_preds = pipe.predict(test_df)
train_mse = mean_squared_error(y_train, train_preds)
test_mse = mean_squared_error(y_test, test_preds)

# Answer check
print(f'Train MSE: {train_mse: .2f}')
print(f'Test MSE: {test_mse: .2f}')
pipe
# %%

# %% md
# [Back to top](#Index)
# 
# #### Problem 5
# 
# #### Backward Selction
# 
# **Points: 20**
# 
# Similar to the forward selection method, backward selection starts with all features and sequentially eliminates features until the threshold is achieved.  Use the selector `backward_selector` below to again build a pipeline named `backward_pipe` and fit a `LinearRegression` model using three features from `train_df`. 
# 
# Assign the train and test mean squared errors as `backward_train_mse` and `backward_test_mse` respectively.
# %%
backward_selector = SequentialFeatureSelector(LinearRegression(),
                                              n_features_to_select=3,
                                              direction='backward')
# %%
### GRADED
backward_pipe = ''
backward_train_mse = ''
backward_test_mse = ''

# YOUR CODE HERE
backward_pipe = Pipeline([('column_selector', backward_selector), ('linreg', LinearRegression())])
backward_pipe.fit(train_df, y_train)
train_preds = backward_pipe.predict(train_df)
test_preds = backward_pipe.predict(test_df)
backward_train_mse = mean_squared_error(y_train, train_preds)
backward_test_mse = mean_squared_error(y_test, test_preds)

# Answer check
print(f'Train MSE: {backward_train_mse: .2f}')
print(f'Test MSE: {backward_test_mse: .2f}')
backward_pipe
# %%

# %% md
# #### Further Exploration
# 
# As an optional exercise work on incorporating the `PolynomialFeatures` into the pipeline along with a `TransformedTargetRegressor` to further abstract the modeling process.  
# %%
from sklearn.compose import TransformedTargetRegressor

trans_poly_pipe = Pipeline(
    [('cub_features', PolynomialFeatures(degree=3, include_bias=False)), ('targ_reg', TransformedTargetRegressor())])
trans_poly_pipe.fit(train_df, y_train)
train_preds_cu = trans_poly_pipe.predict(train_df)
test_preds_cu = trans_poly_pipe.predict(test_df)
trans_poly_train_mse = mean_squared_error(y_train, train_preds)
trans_poly_test_mse = mean_squared_error(y_test, test_preds)
print(f'Train MSE: {trans_poly_train_mse: .2f}')
print(f'Test MSE: {trans_poly_test_mse: .2f}')
trans_poly_pipe
# %%
