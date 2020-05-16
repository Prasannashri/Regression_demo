#import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing Dataset
 dataset = pd.read_csv('position_salaries.csv')
 iv  = dataset.iloc[:,1:-1].values #independent variables
 dv = dataset.iloc[:,2].values #dependent variable
  #fitting Linear Model
  from sklearn.linear_model  import LinearRegression
  lin_reg = LinearRegression()
  lin_reg.fit(iv,dv)
  #fiittting Poly  Regression
  from  sklearn.preprocessing import PolynomialFeatures
  poly_reg = PolynomialFeatures(degree=2)
  iv_poly  = poly_reg.fit_transform(iv)
  lin_reg1 = LinearRegression()
  lin_reg1.fit(iv_poly,dv)
  #predic lin_Reg
  ans = lin_reg.predict(iv)
  #predicting poly_reg
  res = lin_reg1.predict(poly_reg.fit_transform(iv))
  