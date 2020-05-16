import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('position_Salaries.csv')
iv = dataset.iloc[:,1:2].values
dv = dataset.iloc[:,2:3].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(iv,dv)
sal_pred = regressor.predict(iv)
sal_pred = regressor.predict(np.array([[6.5]]))
# Visualising the Decision Tree Regression results (higher resolution)
iv_grid = np.arange(min(iv), max(iv), 0.01)
iv_grid = iv_grid.reshape((len(iv_grid), 1))
plt.scatter(iv, dv, color = 'red')
plt.plot(iv_grid, regressor.predict(iv_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()