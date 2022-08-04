# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 05:47:14 2022

@author: mehra
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Encoding categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
#x[:,3] = labelencoder_x.fit_transform(x[:,3])
onehotencoder = OneHotEncoder()
x= np.append(onehotencoder.fit_transform(x[:,[3]]).toarray(), x,1)
x = np.delete(x, 6, axis=1)
x = x[:,[1,2,3,4,5]]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Trainig Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

reg = regressor.fit(x_train, y_train)

# Predictign the test set results

y_pred = pd.DataFrame(regressor.predict(x_test))


# bilding a optimal model using backward elimination

import statsmodels.regression.linear_model as sm
#import statsmodels.api as sm
x = np.append(np.ones((50,1)).astype(int), x,1)


x_opt = np.array(x[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_ols=sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = np.array(x[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_ols=sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = np.array(x[:, [0, 3, 4, 5]], dtype=float)
regressor_ols=sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()


x_opt = np.array(x[:, [0, 3, 5]], dtype=float)
regressor_ols=sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = np.array(x[:, [0, 3]], dtype=float)
regressor_ols=sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

