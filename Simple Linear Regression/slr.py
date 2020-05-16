#import librarires
import numpy as np
import matplotlib.pyplot as mlt
import pandas as pd
# preprocess the dataset
dataset = pd.read_csv('Salary_Data.csv')
iv = dataset.iloc[:,:-1].values
dv = dataset.iloc[:,1].values
 #spliting
from sklearn.model_selection import train_test_split
iv_train,iv_test,dv_train,dv_test = train_test_split(iv,dv,test_size=1/3,random_state=0)
 #fiting model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(iv_train,dv_train)
#prediction
sal_pred = regressor.predict(iv_test)
 # Visualising the Training set results
mlt.scatter(iv_train,dv_train,color='red')
mlt.plot(iv_train,regressor.predict(iv_train),color='blue')
mlt.title('Sal vs YOE')
mlt.xlabel('Years of  Exp')
mlt.ylabel('Sal')
mlt.show()

 
