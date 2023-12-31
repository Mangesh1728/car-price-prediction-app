# -*- coding: utf-8 -*-
"""Project Car price prediction 4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KuqiaejUuw7LiifHL7l0UzNJNlv_uWxg

ASSIGNMENT 4

Consider the car_price prediction problem:
1.Perform EDA.
2.Apply all the regression techniques that you have read till date.
3.Build best model among them and apply hyperparameter tuning on all algo using gridsearchCV.
"""

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
import warnings
warnings.filterwarnings('ignore')

"""Exploratory Data Analysis (EDA)"""

df = pd.read_csv('CarPrice.csv')
df.head()

df.shape

df.columns

print(df.info())

df.describe()

df.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Set up the figure and axis grid
fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(15, 15))
fig.subplots_adjust(hspace=0.5)

# Plot histograms and kernel density plots for numerical features
for i, col in enumerate(df.columns[:-1]):
    sns.histplot(df[col], kde=True, ax=axes[i//4, i%4])
    axes[i//4, i%4].set_title(col)

plt.show()

# Set up the figure and axis grid
fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(15, 15))
fig.subplots_adjust(hspace=0.5)

# Plot box plots for numerical features with hue
for i, col in enumerate(df.columns[:-1]):
    sns.boxplot(x='dummy', y=df[col], data=df.assign(dummy=1), ax=axes[i//4, i%4])
    axes[i//4, i%4].set_title(col)
    axes[i//4, i%4].set(xlabel='')  # remove x-axis label

plt.show()

"""Bivariate Analysis:
Explore relationships between features and the target variable (price) using scatter plots or correlation matrices.
"""

import seaborn as sns
import matplotlib.pyplot as plt

# Set up the figure and axis grid
fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(15, 15))
fig.subplots_adjust(hspace=0.5)

# Plot scatter plots for numerical features against the target variable (price)
for i, col in enumerate(df.columns[:-1]):
    sns.scatterplot(x=df[col], y=df['price'], ax=axes[i//4, i%4])
    axes[i//4, i%4].set_title(col)

plt.show()

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Matrix")
plt.show()

"""Categorical Variables:
Use count plots or bar plots to analyze distribution for categorical variables.
"""

import seaborn as sns
import matplotlib.pyplot as plt

# Set up the figure and axis grid
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
fig.subplots_adjust(hspace=0.5)

# Plot count plots for categorical variables
for i, col in enumerate(df.select_dtypes(include='object').columns):
    sns.countplot(x=col, data=df, ax=axes[i//4, i%4])
    axes[i//4, i%4].set_title(col)

plt.show()

"""Linear Regression:
Start with simple linear regression.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Assume 'horsepower' is a feature for simple linear regression
X = df[['horsepower']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Plot the regression line
plt.scatter(X_test, y_test, color='black', label='Actual Data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression Line')
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

"""Ridge Regression and Lasso Regression:
•	Regularized linear regression to handle multicollinearity

"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Assume multiple features, for example: 'horsepower', 'curbweight', and 'enginesize'
X = df[['horsepower', 'curbweight', 'enginesize']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)  # You can adjust the alpha (regularization strength)
ridge_model.fit(X_train, y_train)
ridge_y_pred = ridge_model.predict(X_test)

# Lasso Regression
lasso_model = Lasso(alpha=1.0)  # You can adjust the alpha (regularization strength)
lasso_model.fit(X_train, y_train)
lasso_y_pred = lasso_model.predict(X_test)

# Evaluate Ridge Regression
ridge_mse = mean_squared_error(y_test, ridge_y_pred)
ridge_r2 = r2_score(y_test, ridge_y_pred)
print(f'Ridge Regression - Mean Squared Error: {ridge_mse:.2f}, R-squared: {ridge_r2:.2f}')

# Evaluate Lasso Regression
lasso_mse = mean_squared_error(y_test, lasso_y_pred)
lasso_r2 = r2_score(y_test, lasso_y_pred)
print(f'Lasso Regression - Mean Squared Error: {lasso_mse:.2f}, R-squared: {lasso_r2:.2f}')

"""Decision Tree Regression:
•	Non-linear regression model

"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Assume multiple features, for example: 'horsepower', 'curbweight', and 'enginesize'
X = df[['horsepower', 'curbweight', 'enginesize']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Regressor
tree_model = DecisionTreeRegressor(max_depth=5)  # You can adjust the max_depth parameter
tree_model.fit(X_train, y_train)

# Make predictions on the test data
tree_y_pred = tree_model.predict(X_test)

# Evaluate the model
tree_mse = mean_squared_error(y_test, tree_y_pred)
tree_r2 = r2_score(y_test, tree_y_pred)
print(f'Decision Tree Regression - Mean Squared Error: {tree_mse:.2f}, R-squared: {tree_r2:.2f}')

# Plot the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(tree_model, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Regression')
plt.show()
# Plot the predicted vs. actual prices
plt.scatter(y_test, tree_y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Decision Tree Regression - Predicted vs. Actual Prices')
plt.show()

"""Random Forest Regression:
•	Ensemble method for improved accuracy.

"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Assume multiple features, for example: 'horsepower', 'curbweight', and 'enginesize'
X = df[['horsepower', 'curbweight', 'enginesize']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust the number of estimators
rf_model.fit(X_train, y_train)

# Make predictions on the test data
rf_y_pred = rf_model.predict(X_test)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)
print(f'Random Forest Regression - Mean Squared Error: {rf_mse:.2f}, R-squared: {rf_r2:.2f}')

# Visualize the importance of features
feature_importance = rf_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importance[sorted_idx], align="center")
plt.xticks(range(X.shape[1]), X.columns[sorted_idx], rotation=45)
plt.title("Random Forest Feature Importances")
plt.show()

"""Support Vector Regression (SVR):
•	Useful for handling non-linear relationships.

"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Assume multiple features, for example: 'horsepower', 'curbweight', and 'enginesize'
X = df[['horsepower', 'curbweight', 'enginesize']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Support Vector Regressor
svr_model = SVR(kernel='linear')  # You can try different kernels like 'rbf' for non-linear relationships
svr_model.fit(X_train, y_train)

# Make predictions on the test data
svr_y_pred = svr_model.predict(X_test)

# Evaluate the model
svr_mse = mean_squared_error(y_test, svr_y_pred)
svr_r2 = r2_score(y_test, svr_y_pred)
print(f'Support Vector Regression - Mean Squared Error: {svr_mse:.2f}, R-squared: {svr_r2:.2f}')

# Visualize the predictions with the regression line
plt.scatter(y_test, svr_y_pred, label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', label='Regression Line')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Support Vector Regression - Predicted vs. Actual Prices')
plt.legend()
plt.show()

"""Gradient Boosting Regression :
•	Boosting algorithms for improved accuracy.

using XGBoost for regression:
"""

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Assume multiple features, for example: 'horsepower', 'curbweight', and 'enginesize'
X = df[['horsepower', 'curbweight', 'enginesize']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)  # You can adjust hyperparameters
xgb_model.fit(X_train, y_train)

# Make predictions on the test data
xgb_y_pred = xgb_model.predict(X_test)

# Evaluate the model
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_r2 = r2_score(y_test, xgb_y_pred)
print(f'XGBoost Regression - Mean Squared Error: {xgb_mse:.2f}, R-squared: {xgb_r2:.2f}')

# Visualize the predictions with the regression line
plt.scatter(y_test, xgb_y_pred, label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', label='Regression Line')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('XGBoost Regression - Predicted vs. Actual Prices')
plt.legend()
plt.show()

"""LightGBM Regression:"""

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Assume multiple features, for example: 'horsepower', 'curbweight', and 'enginesize'
X = df[['horsepower', 'curbweight', 'enginesize']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LightGBM Regressor
lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)  # You can adjust hyperparameters
lgb_model.fit(X_train, y_train)

# Make predictions on the test data
lgb_y_pred = lgb_model.predict(X_test)

# Evaluate the model
lgb_mse = mean_squared_error(y_test, lgb_y_pred)
lgb_r2 = r2_score(y_test, lgb_y_pred)
print(f'LightGBM Regression - Mean Squared Error: {lgb_mse:.2f}, R-squared: {lgb_r2:.2f}')

# Visualize the predictions with the regression line
plt.scatter(y_test, lgb_y_pred, label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', label='Regression Line')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('LightGBM Regression - Predicted vs. Actual Prices')
plt.legend()
plt.show()

"""Model Building:
Build models using the above techniques and evaluate their performance using metrics such as Mean Squared Error (MSE), R-squared error.

"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Assume multiple features, for example: 'horsepower', 'curbweight', and 'enginesize'
X = df[['horsepower', 'curbweight', 'enginesize']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_y_pred = linear_model.predict(X_test)
linear_mse = mean_squared_error(y_test, linear_y_pred)
linear_r2 = r2_score(y_test, linear_y_pred)
print(f'Linear Regression - Mean Squared Error: {linear_mse:.2f}, R-squared: {linear_r2:.2f}')

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_y_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_y_pred)
ridge_r2 = r2_score(y_test, ridge_y_pred)
print(f'Ridge Regression - Mean Squared Error: {ridge_mse:.2f}, R-squared: {ridge_r2:.2f}')

# Lasso Regression
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)
lasso_y_pred = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_y_pred)
lasso_r2 = r2_score(y_test, lasso_y_pred)
print(f'Lasso Regression - Mean Squared Error: {lasso_mse:.2f}, R-squared: {lasso_r2:.2f}')

# Decision Tree Regression
tree_model = DecisionTreeRegressor(max_depth=5)
tree_model.fit(X_train, y_train)
tree_y_pred = tree_model.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_y_pred)
tree_r2 = r2_score(y_test, tree_y_pred)
print(f'Decision Tree Regression - Mean Squared Error: {tree_mse:.2f}, R-squared: {tree_r2:.2f}')

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)
print(f'Random Forest Regression - Mean Squared Error: {rf_mse:.2f}, R-squared: {rf_r2:.2f}')

# Support Vector Regression (SVR)
svr_model = SVR(kernel='linear')
svr_model.fit(X_train, y_train)
svr_y_pred = svr_model.predict(X_test)
svr_mse = mean_squared_error(y_test, svr_y_pred)
svr_r2 = r2_score(y_test, svr_y_pred)
print(f'Support Vector Regression - Mean Squared Error: {svr_mse:.2f}, R-squared: {svr_r2:.2f}')

# XGBoost Regression
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_r2 = r2_score(y_test, xgb_y_pred)
print(f'XGBoost Regression - Mean Squared Error: {xgb_mse:.2f}, R-squared: {xgb_r2:.2f}')

# LightGBM Regression
lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_y_pred = lgb_model.predict(X_test)
lgb_mse = mean_squared_error(y_test, lgb_y_pred)
lgb_r2 = r2_score(y_test, lgb_y_pred)
print(f'LightGBM Regression - Mean Squared Error: {lgb_mse:.2f}, R-squared: {lgb_r2:.2f}')

# Visualize the predictions with the regression line for XGBoost Regression
plt.scatter(y_test, xgb_y_pred, label='XGBoost Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', label='Regression Line')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

"""Hyperparameter Tuning:
Use GridSearchCV to perform hyperparameter tuning on the best-performing models.

"""

from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Assume multiple features, for example: 'horsepower', 'curbweight', and 'enginesize'
X = df[['horsepower', 'curbweight', 'enginesize']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Create an XGBoost Regressor
xgb_model = XGBRegressor(random_state=42)

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding MSE for XGBoost Regression
best_params = grid_search.best_params_
best_mse = -grid_search.best_score_
best_xgb_model = grid_search.best_estimator_

# Make predictions on the test data using the best model
best_xgb_y_pred = best_xgb_model.predict(X_test)

# Evaluate the best model
best_xgb_mse = mean_squared_error(y_test, best_xgb_y_pred)
best_xgb_r2 = r2_score(y_test, best_xgb_y_pred)

# Create a DataFrame for better visualization
results_df = pd.DataFrame({
    'Parameter': ['learning_rate', 'max_depth', 'n_estimators', 'Best Mean Squared Error', 'Best XGBoost MSE', 'Best XGBoost R-squared'],
    'Value': [best_params['learning_rate'], best_params['max_depth'], best_params['n_estimators'], best_mse, best_xgb_mse, best_xgb_r2]
})

# Print the DataFrame
print(results_df)

"""Model Evaluation:
After hyperparameter tuning, reevaluate the models and compare their performance. Choose the best-performing model based on your evaluation metrics.

"""

# Re-evaluate the XGBoost model after hyperparameter tuning
best_xgb_y_pred = best_xgb_model.predict(X_test)

# Evaluate the best model
best_xgb_mse = mean_squared_error(y_test, best_xgb_y_pred)
best_xgb_r2 = r2_score(y_test, best_xgb_y_pred)

# Print the evaluation metrics for the best XGBoost model
print(f'Best XGBoost Regression - Mean Squared Error: {best_xgb_mse:.2f}, R-squared: {best_xgb_r2:.2f}')

# Compare with the untuned XGBoost model for reference
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_r2 = r2_score(y_test, xgb_y_pred)

# Print the evaluation metrics for the untuned XGBoost model
print(f'Untuned XGBoost Regression - Mean Squared Error: {xgb_mse:.2f}, R-squared: {xgb_r2:.2f}')

"""Final Model:
Train chosen model on the entire dataset and save it for future predictions

"""

# Combine the entire dataset for training the final model
X_final = pd.concat([X_train, X_test], axis=0)
y_final = pd.concat([y_train, y_test], axis=0)

# Train the final XGBoost model on the entire dataset
final_xgb_model = XGBRegressor(**best_params, random_state=42)  # Using the best hyperparameters
final_xgb_model.fit(X_final, y_final)

# Save the final XGBoost model using joblib (you can use any other method to save the model)
import joblib

joblib.dump(final_xgb_model, 'final_xgboost_model.joblib')

print("Final XGBoost model trained on the entire dataset and saved for future predictions.")

! pip install streamlit -q

!wget -q -O - ipv4.icanhazip.com

! streamlit run app.py & npx localtunnel --port 8501