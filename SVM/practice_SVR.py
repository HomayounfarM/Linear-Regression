# SVR

# Importing the librariees
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Import the dataset
for file in os.listdir():
    if file.endswith('.csv'):
        print(file)
        name = file

dataset = pd.read_csv(name)
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))


# Fitting SVR to the dataset
# Create your regressor hear

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)


# predicting a new results
sample = sc_X.transform(np.array([[6.5]]))

y_pred = regressor.predict(sample)

y_pred = sc_y.inverse_transform([y_pred])

# Visualizing the SVR results
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape(-1,1)
plt.scatter(X, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
