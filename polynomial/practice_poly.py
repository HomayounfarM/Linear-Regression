# Data Preprocessing Template

# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


# Splitting the dataset into the Training set and Test set
'''from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state=0)'''

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(X, y)

import statsmodels.regression.linear_model as sm
X_temp = np.append(np.ones((len(X),1)).astype(int), X,1)
regressor_ols=sm.OLS(endog = y, exog = X_temp).fit()
regressor_ols.summary()

# Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing the linear regression results
plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('truth or bluff')
plt.xlabel('position level')
plt.ylabel('Salary')

# Visualizing the poynomial regression results
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'green')
plt.title('truth or bluff')
plt.xlabel('position level')
plt.ylabel('Salary')

plt.show()

# Predicting a new result with linear regression
sample = np.array(6.5)
sample = sample.reshape(-1,1)

lin_reg.predict(sample)

# Predicting a new result with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(sample))











































