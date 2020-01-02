# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 09:16:48 2019

@author: RAJ KAMAL B GOWDA
"""

#importing the libraries and data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline

Train_data=pd.read_csv('C:/Users/RANVI/Desktop/6 Benches/Linear regression/silfra/3.Cleaned output data/cleaned_train_data.csv')
Test_data=pd.read_csv('C:/Users/RANVI/Desktop/6 Benches/Linear regression/silfra/3.Cleaned output data/cleaned_test_data.csv')


#Making x and y separation in both train and test data

#      train
train_data_copy=Train_data.copy()

y_original_train = train_data_copy['SalePrice']

del train_data_copy['SalePrice']
x_original_train= train_data_copy.copy()

#      test

x_original_test=Test_data


#Selecting good features for good performance
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2,k=10)
fit= bestfeatures.fit(x_original_train,y_original_train)

dfscores=pd.DataFrame({'Scores':fit.scores_})
dfcolumns=pd.DataFrame({'Features':x_original_train.columns})
feature_scores= pd.concat([dfcolumns,dfscores],axis=1)

Best_x_features_scores=(feature_scores.nlargest(35,'Scores'))

Best_x_features_scores.iloc[:,0]

Best_x_features_scores['Features'].values

#Now select These Features to train the model
Xtr=x_original_train[Best_x_features_scores['Features'].values]
Ytr=y_original_train


# Fitting Multiple Linear Regression to the Training set with Stat Model
import statsmodels.api as sm
X_train_sm = sm.add_constant(Xtr)

# Run model (Model training)
model = sm.OLS(Ytr, X_train_sm).fit()

y_train_predicted = model.predict(X_train_sm) # make the predictions by the model


from sklearn import metrics
from sklearn.metrics import r2_score

MAE=( metrics.mean_absolute_error(Ytr, y_train_predicted))
MSE=(metrics.mean_squared_error(Ytr, y_train_predicted))
RMSE= (np.sqrt(metrics.mean_squared_error(Ytr, y_train_predicted)))
r2_value=r2_score(Ytr,y_train_predicted)

r2_value,RMSE

# Print out the statistics
model.summary()


#delete un signigicant model

del X_train_sm['LotConfig_FR3']
del X_train_sm['SaleCondition_Alloca']
del X_train_sm['LotFrontage']
del X_train_sm['Exterior1st_ImStucc']
del X_train_sm['2ndFlrSF']
del X_train_sm['LotShape_IR3']
del X_train_sm['SaleType_Oth']
del X_train_sm['BsmtUnfSF']
del X_train_sm['SaleCondition_Partial']
del X_train_sm['SaleType_Con']
del X_train_sm['GarageCars_3']
del X_train_sm['1stFlrSF']
del X_train_sm['Condition2_RRAn']
del X_train_sm['MasVnrArea']
del X_train_sm['RoofMatl_Membran']
del X_train_sm['Neighborhood_CollgCr']
del X_train_sm['BsmtFinSF2']
del X_train_sm['Neighborhood_Veenker']
del X_train_sm['Neighborhood_Gilbert']
del X_train_sm['SaleType_New']
del X_train_sm['Condition2_PosN']
del X_train_sm['Neighborhood_Somerst']


#Prediction for the test data

A= set(X_train_sm.columns)
B=set(x_original_test.columns)


Matched_cols = list(A & B)
Xtst=x_original_test[Matched_cols]


# Fitting Multiple Linear Regression to the Test set with Stat Model

X_test_sm = sm.add_constant(Xtst)
type(X_test_sm)

X_train_sm

Xtest_in_order_of_x_train=X_test_sm[X_train_sm.columns]

X_test=Xtest_in_order_of_x_train

#Prediction for X test
y_test_predicted = model.predict(X_test) # make the predictions by the model

y_test_predicted






