#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing Dataset
dataset = pd.read_csv('position_salaries.csv')
iv = dataset.iloc[:,1:2].values
dv = dataset.iloc[:,2:3].values
#fitting model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(iv,dv)
#prediction
sal_pred = regressor.predict(np.array([[6.5]]))
