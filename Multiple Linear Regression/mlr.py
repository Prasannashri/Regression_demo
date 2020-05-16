import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing Dataset
 dataset = pd.read_csv('50_startups.csv')
 iv  = dataset.iloc[:,:-1].values 
 dv = dataset.iloc[:,4].values #dependent variable
 #Encoding Categorical data
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    labelencoder_iv = LabelEncoder()
    iv[:,3]=labelencoder_iv.fit_transform(iv[:,3]) 
    onehotencoder = OneHotEncoder(categorical_features = [3])
    iv = onehotencoder.fit_transform(iv).toarray()
 #avoiding Dummy Variable Trap
   iv = iv[:,1:]
   #splitting 
    from sklearn.model_selection import train_test_split
    iv_train,iv_test,dv_train,dv_test = train_test_split(iv,dv,test_size=0.2,random_state=0)
    #fitting model
    from sklearn.linear_model  import LinearRegression
    regressor = LinearRegression()
    regressor.fit(iv_train,dv_train)
    #prediction
    prof_pred = regressor.predict(iv_test)
