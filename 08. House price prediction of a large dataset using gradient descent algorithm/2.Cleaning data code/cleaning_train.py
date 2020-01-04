# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:13:29 2019

@author: RAJ KAMAL B GOWDA
"""
#import all the libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline

#read the train dataset

df=pd.read_csv("C:/Users/RANVI/Desktop/Silfra assignment- Rajkamal/1.Original data/train.csv")

#Seperate numerical and categorical data

df.columns #all columns
all_cols=set(df.columns)

numerical_cols=set(df._get_numeric_data().columns) #numerical columns
len_numerical_cols=len(numerical_cols)


non_numerical_cols = all_cols-numerical_cols #non numerical columns
len_non_numerical_cols=len(non_numerical_cols)


#cross_check
43+38==81  #See true or  false

#Exploratory data analysis and cleaning of the data
df_eda=df.copy()

#Exploratory data analysis and cleaning of the data

#0 ID
df_eda.columns              #unnecessary column
plt.hist(df_eda['Id'])
del df_eda['Id']


#1 MSSubclass
plt.hist(df_eda['MSSubClass']) #catogorical and Numerical
sns.kdeplot(df_eda['MSSubClass'])

df_eda['MSSubClass'].isnull().sum()  #no null values

plt.boxplot(df_eda['MSSubClass']) #to see outliers

percentiles = df_eda['MSSubClass'].quantile([0.1,0.9]).values #outliers treatment
df_eda['MSSubClass']=df_eda['MSSubClass'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['MSSubClass'])

df_eda['MSSubClass'].value_counts().count() #To see how many categories

dfDummies_MSSubClass = pd.get_dummies(df_eda['MSSubClass'], prefix = 'MSSubClass') #one hat encoding

del df_eda['MSSubClass'] #del column in original data

del dfDummies_MSSubClass['MSSubClass_20'] #del one dummy variable



#2 MSZoning
plt.hist(df_eda['MSZoning']) #categorical and non numerical

df_eda['MSZoning'].isnull().sum() #no null values
df_eda['MSZoning'] = df_eda['MSZoning'].fillna((df_eda['MSZoning'].mode())) 

df_eda['MSZoning'].value_counts().count()


dfDummies_MSZoning = pd.get_dummies(df_eda['MSZoning'], prefix = 'MSZoning') #one hat encoding

del df_eda['MSZoning'] #del column in original data

del dfDummies_MSZoning['MSZoning_FV'] #del one dummy variable



#3 'LotFrontage'
plt.hist(df_eda['LotFrontage']) #continuous and numerical
sns.kdeplot(df_eda['LotFrontage'])


df_eda['LotFrontage'].isnull().sum() #check for null values #there are null values
df_eda['LotFrontage'] = df_eda['LotFrontage'].fillna((df_eda['LotFrontage'].median()))  

df_eda['LotFrontage'].isnull().sum()

plt.boxplot(df_eda['LotFrontage'])

percentiles = df_eda['LotFrontage'].quantile([0.1,0.9]).values #outliers treatment
df_eda['LotFrontage']=df_eda['LotFrontage'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['LotFrontage'])

#4 'LotArea'
plt.hist(df_eda['LotArea'])   #continuous and numerical
sns.kdeplot(df_eda['LotArea'])


df_eda['LotArea'].isnull().sum() #check for null values #there are null values
df_eda['LotArea'] = df_eda['LotArea'].fillna((df_eda['LotArea'].median())) 
df_eda['LotArea'].isnull().sum()

plt.boxplot(df_eda['LotArea'])

percentiles = df_eda['LotArea'].quantile([0.1,0.9]).values #outliers treatment
df_eda['LotArea']=df_eda['LotArea'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['LotArea'])


#5  'Street'

df_eda['Street']

plt.hist(df_eda['Street'])   #categorical and non numerical

df_eda['Street'].isnull().sum()
df_eda['Street'] = df_eda['Street'].fillna((df_eda['Street'].mode())) 
df_eda['Street'].isnull().sum()


dfDummies_Street = pd.get_dummies(df_eda['Street'], prefix = 'Street') #one hat encoding

del df_eda['Street'] #delete original
del dfDummies_Street['Street_Grvl'] #delete dummy

#6 Alley

df_eda['Alley']
df_eda['Alley'].count()
del  df_eda['Alley']


#7 LotShape

df_eda['LotShape'].describe()
df_eda['LotShape'].count()

plt.hist(df_eda['LotShape']) #categorical and non numerical

df_eda['LotShape'].isnull().sum()
df_eda['LotShape'] = df_eda['LotShape'].fillna((df_eda['LotShape'].mode())) 
df_eda['LotShape'].isnull().sum()


dfDummies_LotShape = pd.get_dummies(df_eda['LotShape'], prefix = 'LotShape') #one hat encoding

del df_eda['LotShape'] #delete original

del dfDummies_LotShape['LotShape_IR1'] #delete dummy


#8 LandContour

df_eda['LandContour'].describe()
df_eda['LandContour'].count()

plt.hist(df_eda['LandContour']) #categorical and non numerical

df_eda['LandContour'].isnull().sum()
df_eda['LandContour'] = df_eda['LandContour'].fillna((df_eda['LandContour'].mode())) 
df_eda['LandContour'].isnull().sum()


dfDummies_LandContour = pd.get_dummies(df_eda['LandContour'], prefix = 'LandContour') #one hat encoding

del df_eda['LandContour'] #delete original

del dfDummies_LandContour['LandContour_Bnk'] #delete dummy


#9 Utilities

df.columns[9]

df_eda['Utilities']
df_eda['Utilities'].count()

plt.hist(df_eda['Utilities']) #categorical and non numerical

df_eda['Utilities'].isnull().sum()
df_eda['Utilities'] = df_eda['Utilities'].fillna((df_eda['Utilities'].mode())) 
df_eda['Utilities'].isnull().sum()


dfDummies_Utilities = pd.get_dummies(df_eda['Utilities'], prefix = 'Utilities') #one hat encoding

del df_eda['Utilities'] #delete original

del dfDummies_Utilities['Utilities_AllPub'] #delete dummy

#10 LotConfig
df.columns[10]

df_eda['LotConfig']
df_eda['LotConfig'].count()

plt.hist(df_eda['LotConfig']) #categorical and non numerical

df_eda['LotConfig'].isnull().sum()
df_eda['LotConfig'] = df_eda['LotConfig'].fillna((df_eda['LotConfig'].mode())) 
df_eda['LotConfig'].isnull().sum()


dfDummies_LotConfig = pd.get_dummies(df_eda['LotConfig'], prefix = 'LotConfig') #one hat encoding

del df_eda['LotConfig'] #delete original

del dfDummies_LotConfig['LotConfig_Corner'] #delete dummy


#11 LandSlope
df.columns[11]

df_eda['LandSlope']
df_eda['LandSlope'].count()

plt.hist(df_eda['LandSlope']) #categorical and non numerical

df_eda['LandSlope'].isnull().sum()
df_eda['LandSlope'] = df_eda['LandSlope'].fillna((df_eda['LandSlope'].mode())) 
df_eda['LandSlope'].isnull().sum()


dfDummies_LandSlope = pd.get_dummies(df_eda['LandSlope'], prefix = 'LandSlope') #one hat encoding

del df_eda['LandSlope'] #delete original

del dfDummies_LandSlope['LandSlope_Gtl'] #delete dummy


#12 
df.columns[12]

df_eda['Neighborhood']
df_eda['Neighborhood'].describe()
df_eda['Neighborhood'].count()

plt.hist(df_eda['Neighborhood'])  #categorical and non numerical

df_eda['Neighborhood'].isnull().sum()
df_eda['Neighborhood'] = df_eda['Neighborhood'].fillna((df_eda['Neighborhood'].mode())) 
df_eda['Neighborhood'].isnull().sum()


dfDummies_Neighborhood = pd.get_dummies(df_eda['Neighborhood'], prefix = 'Neighborhood') #one hat encoding

del df_eda['Neighborhood'] #delete original

del dfDummies_Neighborhood['Neighborhood_Blmngtn'] #delete dummy


#13 Condition1

df.columns[13]

df_eda['Condition1']
df_eda['Condition1'].describe()
df_eda['Condition1'].count()

plt.hist(df_eda['Condition1']) #categorical_and_non_numerical

df_eda['Condition1'].isnull().sum()
df_eda['Condition1'] = df_eda['Condition1'].fillna((df_eda['Condition1'].mode())) 
df_eda['Condition1'].isnull().sum()


dfDummies_Condition1= pd.get_dummies(df_eda['Condition1'], prefix = 'Condition1') #one hat encoding

del df_eda['Condition1'] #delete original

del dfDummies_Condition1['Condition1_Artery'] #delete dummy

#14
df.columns[14]

df_eda['Condition2']
df_eda['Condition2'].describe()
df_eda['Condition2'].count()

plt.hist(df_eda['Condition2']) #categorical_and_non_numerical

df_eda['Condition2'].isnull().sum()
df_eda['Condition2'] = df_eda['Condition2'].fillna((df_eda['Condition2'].mode())) 
df_eda['Condition2'].isnull().sum()


dfDummies_Condition2= pd.get_dummies(df_eda['Condition2'], prefix = 'Condition2') #one hat encoding

del df_eda['Condition2'] #delete original

del dfDummies_Condition2['Condition2_Artery'] #delete dummy

#15
df.columns[15]

df_eda['BldgType']
df_eda['BldgType'].describe()
df_eda['BldgType'].count()

plt.hist(df_eda['BldgType']) #categorical_and_non_numerical

df_eda['BldgType'].isnull().sum()
df_eda['BldgType'] = df_eda['BldgType'].fillna((df_eda['BldgType'].mode())) 
df_eda['BldgType'].isnull().sum()


dfDummies_BldgType= pd.get_dummies(df_eda['BldgType'], prefix = 'BldgType') #one hat encoding

del df_eda['BldgType'] #delete original

del dfDummies_BldgType['BldgType_1Fam'] #delete dummy


#16 HouseStyle
df.columns[16]

df_eda['HouseStyle']
df_eda['HouseStyle'].describe()
df_eda['HouseStyle'].count()

plt.hist(df_eda['HouseStyle']) #categorical_and_non_numerical

df_eda['HouseStyle'].isnull().sum()
df_eda['HouseStyle'] = df_eda['HouseStyle'].fillna((df_eda['HouseStyle'].mode())) 
df_eda['HouseStyle'].isnull().sum()


dfDummies_HouseStyle= pd.get_dummies(df_eda['HouseStyle'], prefix = 'HouseStyle') #one hat encoding

del df_eda['HouseStyle'] #delete original

del dfDummies_HouseStyle['HouseStyle_1.5Fin'] #delete dummy

#17 OverallQual
df.columns[17]

df_eda['OverallQual'].unique()
df_eda['OverallQual'].describe()
df_eda['OverallQual'].count()

plt.hist(df_eda['OverallQual']) #categorical and numerical

sns.kdeplot(df_eda['OverallQual'])

df_eda['OverallQual'].isnull().sum()  #no null values
df_eda['OverallQual'] = df_eda['OverallQual'].fillna((df_eda['OverallQual'].mode())) 
df_eda['OverallQual'].isnull().sum()

plt.boxplot(df_eda['OverallQual']) #to see outliers

percentiles = df_eda['OverallQual'].quantile([0.1,0.9]).values #outliers treatment
df_eda['OverallQual']=df_eda['OverallQual'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['OverallQual'])

dfDummies_OverallQual = pd.get_dummies(df_eda['OverallQual'], prefix = 'OverallQual') #one hat encoding

del df_eda['OverallQual'] #del column in original data

del dfDummies_OverallQual['OverallQual_5'] #del one dummy variable

#18 OverallCond
df.columns[18]

df_eda['OverallCond'].unique()
df_eda['OverallCond'].describe()
df_eda['OverallCond'].count()

plt.hist(df_eda['OverallCond'])   #categorical and numerical

sns.kdeplot(df_eda['OverallCond'])

df_eda['OverallCond'].isnull().sum()  #no null values
df_eda['OverallCond'] = df_eda['OverallCond'].fillna((df_eda['OverallCond'].mode())) 
df_eda['OverallCond'].isnull().sum()

plt.boxplot(df_eda['OverallCond']) #to see outliers

percentiles = df_eda['OverallCond'].quantile([0.1,0.9]).values #outliers treatment
df_eda['OverallCond']=df_eda['OverallCond'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['OverallCond'])

dfDummies_OverallCond = pd.get_dummies(df_eda['OverallCond'], prefix = 'OverallCond') #one hat encoding

del df_eda['OverallCond'] #del column in original data

del dfDummies_OverallCond['OverallCond_5'] #del one dummy variable


#19 YearBuilt
df.columns[19]

df_eda['YearBuilt'].unique()
df_eda['YearBuilt'].describe()
df_eda['YearBuilt'].count()

plt.hist(df_eda['YearBuilt']) #continuous and numerical
sns.kdeplot(df_eda['YearBuilt'])


df_eda['YearBuilt'].isnull().sum() #check for null values #there are null values
df_eda['YearBuilt'] = df_eda['YearBuilt'].fillna((df_eda['YearBuilt'].mode()))  #since it is year fill with mode
df_eda['YearBuilt'].isnull().sum()

plt.boxplot(df_eda['YearBuilt'])

percentiles = df_eda['YearBuilt'].quantile([0.1,0.9]).values #outliers treatment
df_eda['YearBuilt']=df_eda['YearBuilt'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['YearBuilt'])



#20 
df.columns[20]

df_eda['YearRemodAdd'].unique()
df_eda['YearRemodAdd'].describe()
df_eda['YearRemodAdd'].count()

plt.hist(df_eda['YearRemodAdd']) #continuous and numerical
sns.kdeplot(df_eda['YearRemodAdd'])


df_eda['YearRemodAdd'].isnull().sum() #check for null values #there are null values
df_eda['YearRemodAdd'] = df_eda['YearRemodAdd'].fillna((df_eda['YearRemodAdd'].mode()))  #since it is year fill with mode
df_eda['YearRemodAdd'].isnull().sum()

plt.boxplot(df_eda['YearRemodAdd'])

percentiles = df_eda['YearRemodAdd'].quantile([0.1,0.9]).values #outliers treatment
df_eda['YearRemodAdd']=df_eda['YearRemodAdd'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['YearRemodAdd'])


#21 RoofStyle
df.columns[21]

df_eda['RoofStyle'].unique()
df_eda['RoofStyle'].describe()
df_eda['RoofStyle'].count()

plt.hist(df_eda['RoofStyle']) #categorical and non numerical

df_eda['RoofStyle'].isnull().sum()
df_eda['RoofStyle'] = df_eda['RoofStyle'].fillna((df_eda['RoofStyle'].mode())) 
df_eda['RoofStyle'].isnull().sum()


dfDummies_RoofStyle= pd.get_dummies(df_eda['RoofStyle'], prefix = 'RoofStyle') #one hat encoding

del df_eda['RoofStyle'] #delete original

del dfDummies_RoofStyle['RoofStyle_Flat'] #delete dummy

#22 RoofMatl
df.columns[22]

df_eda['RoofMatl'].unique()
df_eda['RoofMatl'].describe()
df_eda['RoofMatl'].count()

plt.hist(df_eda['RoofMatl']) #catogorical and non numerical

df_eda['RoofMatl'].isnull().sum()
df_eda['RoofMatl'] = df_eda['RoofMatl'].fillna((df_eda['RoofMatl'].mode())) 
df_eda['RoofMatl'].isnull().sum()


dfDummies_RoofMatl= pd.get_dummies(df_eda['RoofMatl'], prefix = 'RoofMatl') #one hat encoding

del df_eda['RoofMatl'] #delete original

del dfDummies_RoofMatl['RoofMatl_ClyTile'] #delete dummy

#23 Exterior1st
df.columns[23]

df_eda['Exterior1st'].unique()
df_eda['Exterior1st'].describe()
df_eda['Exterior1st'].count()

plt.hist(df_eda['Exterior1st']) #catogorical and non numerical

df_eda['Exterior1st'].isnull().sum()
df_eda['Exterior1st'] = df_eda['Exterior1st'].fillna((df_eda['Exterior1st'].mode())) 
df_eda['Exterior1st'].isnull().sum()


dfDummies_Exterior1st= pd.get_dummies(df_eda['Exterior1st'], prefix = 'Exterior1st') #one hat encoding

del df_eda['Exterior1st'] #delete original

del dfDummies_Exterior1st['Exterior1st_AsbShng'] #delete dummy

#24 Exterior2nd
df.columns[24]

df_eda['Exterior2nd'].unique()
df_eda['Exterior2nd'].describe()
df_eda['Exterior2nd'].count()

plt.hist(df_eda['Exterior2nd']) #catogorical and non numerical

df_eda['Exterior2nd'].isnull().sum()
df_eda['Exterior2nd'] = df_eda['Exterior2nd'].fillna((df_eda['Exterior2nd'].mode())) 
df_eda['Exterior2nd'].isnull().sum()


dfDummies_Exterior2nd= pd.get_dummies(df_eda['Exterior2nd'], prefix = 'Exterior2nd') #one hat encoding

del df_eda['Exterior2nd'] #delete original

del dfDummies_Exterior2nd['Exterior2nd_AsbShng'] #delete dummy

#25 MasVnrType
df.columns[25]

df_eda['MasVnrType'].unique()
df_eda['MasVnrType'].describe()
df_eda['MasVnrType'].count()


df_eda['MasVnrType'].isnull().sum()
df_eda['MasVnrType'] = df_eda['MasVnrType'].fillna((df_eda['MasVnrType'].mode()[0])) 
df_eda['MasVnrType'].isnull().sum()

plt.hist(df_eda['MasVnrType']) #catogorical and non numerical

dfDummies_MasVnrType= pd.get_dummies(df_eda['MasVnrType'], prefix = 'MasVnrType') #one hat encoding

del df_eda['MasVnrType'] #delete original

del dfDummies_MasVnrType['MasVnrType_BrkCmn'] #delete dummy


#26 MasVnrArea
df.columns[26]

df_eda['MasVnrArea'].unique()
df_eda['MasVnrArea'].describe()
df_eda['MasVnrArea'].count()

plt.hist(df_eda['MasVnrArea']) #continuous and numerical
sns.kdeplot(df_eda['MasVnrArea'])


df_eda['MasVnrArea'].isnull().sum() #check for null values #there are null values
df_eda['MasVnrArea'] = df_eda['MasVnrArea'].fillna((df_eda['MasVnrArea'].median()))  #since it is year fill with mode
df_eda['MasVnrArea'].isnull().sum()

plt.boxplot(df_eda['MasVnrArea'])

percentiles = df_eda['MasVnrArea'].quantile([0.1,0.9]).values #outliers treatment
df_eda['MasVnrArea']=df_eda['MasVnrArea'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['MasVnrArea'])

#27 ExterQual
df.columns[27]

df_eda['ExterQual'].unique()
df_eda['ExterQual'].describe()
df_eda['ExterQual'].count()

plt.hist(df_eda['ExterQual']) #categorical and non numerical

df_eda['ExterQual'].isnull().sum()
df_eda['ExterQual'] = df_eda['ExterQual'].fillna((df_eda['ExterQual'].mode())) 
df_eda['ExterQual'].isnull().sum()


dfDummies_ExterQual= pd.get_dummies(df_eda['ExterQual'], prefix = 'ExterQual') #one hat encoding

del df_eda['ExterQual'] #delete original

del dfDummies_ExterQual['ExterQual_Ex'] #delete dummy

#28 ExterCond
df.columns[28]

df_eda['ExterCond'].unique()
df_eda['ExterCond'].describe()
df_eda['ExterCond'].count()

plt.hist(df_eda['ExterCond']) #categorical and non numerical

df_eda['ExterCond'].isnull().sum()
df_eda['ExterCond'] = df_eda['ExterCond'].fillna((df_eda['ExterCond'].mode())) 
df_eda['ExterCond'].isnull().sum()


dfDummies_ExterCond= pd.get_dummies(df_eda['ExterCond'], prefix = 'ExterCond') #one hat encoding

del df_eda['ExterCond'] #delete original

del dfDummies_ExterCond['ExterCond_Ex'] #delete dummy


#29 Foundation
df.columns[29]

df_eda['Foundation'].unique()
df_eda['Foundation'].describe()
df_eda['Foundation'].count()

plt.hist(df_eda['Foundation']) #categorical and non numerical

df_eda['Foundation'].isnull().sum()
df_eda['Foundation'] = df_eda['Foundation'].fillna((df_eda['Foundation'].mode())) 
df_eda['Foundation'].isnull().sum()


dfDummies_Foundation= pd.get_dummies(df_eda['Foundation'], prefix = 'Foundation') #one hat encoding

del df_eda['Foundation'] #delete original

del dfDummies_Foundation['Foundation_BrkTil'] #delete dummy

#30 BsmtQual
df.columns[30]

df_eda['BsmtQual'].unique()
df_eda['BsmtQual'].describe()
df_eda['BsmtQual'].count()

                                      #categorical and non numerical
df_eda['BsmtQual'].isnull().sum()
df_eda['BsmtQual'] = df_eda['BsmtQual'].fillna((df_eda['BsmtQual'].mode()[0])) 
df_eda['BsmtQual'].isnull().sum()

plt.hist(df_eda['BsmtQual'])

dfDummies_BsmtQual= pd.get_dummies(df_eda['BsmtQual'], prefix = 'BsmtQual') #one hat encoding

del df_eda['BsmtQual'] #delete original

del dfDummies_BsmtQual['BsmtQual_Ex'] #delete dummy


#31 BsmtCond
df.columns[31]

df_eda['BsmtCond'].unique() #non numerical and categorical
df_eda['BsmtCond'].describe()
df_eda['BsmtCond'].count()

df_eda['BsmtCond'].isnull().sum()
df_eda['BsmtCond'] = df_eda['BsmtCond'].fillna((df_eda['BsmtCond'].mode()[0])) 
df_eda['BsmtCond'].isnull().sum()

plt.hist(df_eda['BsmtCond'])

dfDummies_BsmtCond= pd.get_dummies(df_eda['BsmtCond'], prefix = 'BsmtCond') #one hat encoding

del df_eda['BsmtCond'] #delete original

del dfDummies_BsmtCond['BsmtCond_Fa'] #delete dummy


#32 BsmtExposure
df.columns[32]

df_eda['BsmtExposure'].unique() #non numerical and categorical
df_eda['BsmtExposure'].describe()
df_eda['BsmtExposure'].count()

df_eda['BsmtExposure'].isnull().sum()
df_eda['BsmtExposure'] = df_eda['BsmtExposure'].fillna((df_eda['BsmtExposure'].mode()[0])) 
df_eda['BsmtExposure'].isnull().sum()

plt.hist(df_eda['BsmtExposure'])

dfDummies_BsmtExposure= pd.get_dummies(df_eda['BsmtExposure'], prefix = 'BsmtExposure') #one hat encoding

del df_eda['BsmtExposure'] #delete original

del dfDummies_BsmtExposure['BsmtExposure_Av'] #delete dummy

#33 BsmtFinType1
df.columns[33]

df_eda['BsmtFinType1'].unique() #non numerical and categorical
df_eda['BsmtFinType1'].describe()
df_eda['BsmtFinType1'].count()

df_eda['BsmtFinType1'].isnull().sum()
df_eda['BsmtFinType1'] = df_eda['BsmtFinType1'].fillna((df_eda['BsmtFinType1'].mode()[0])) 
df_eda['BsmtFinType1'].isnull().sum()

plt.hist(df_eda['BsmtFinType1'])

dfDummies_BsmtFinType1= pd.get_dummies(df_eda['BsmtFinType1'], prefix = 'BsmtFinType1') #one hat encoding

del df_eda['BsmtFinType1'] #delete original

del dfDummies_BsmtFinType1['BsmtFinType1_ALQ'] #delete dummy

#34 BsmtFinSF1
df.columns[34]

df_eda['BsmtFinSF1'].unique()
df_eda['BsmtFinSF1'].describe()
df_eda['BsmtFinSF1'].count()

plt.hist(df_eda['BsmtFinSF1']) #continuous and numerical
sns.kdeplot(df_eda['BsmtFinSF1'])


df_eda['BsmtFinSF1'].isnull().sum() #check for null values #there are null values
df_eda['BsmtFinSF1'] = df_eda['BsmtFinSF1'].fillna((df_eda['BsmtFinSF1'].median()))  #since it is year fill with mode
df_eda['BsmtFinSF1'].isnull().sum()

plt.boxplot(df_eda['BsmtFinSF1'])

percentiles = df_eda['BsmtFinSF1'].quantile([0.1,0.9]).values #outliers treatment
df_eda['BsmtFinSF1']=df_eda['BsmtFinSF1'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['BsmtFinSF1'])

#35
df.columns[35]

df_eda['BsmtFinType2'].unique()
df_eda['BsmtFinType2'].describe()
df_eda['BsmtFinType2'].count()

df_eda['BsmtFinType2'].isnull().sum()
df_eda['BsmtFinType2'] = df_eda['BsmtFinType2'].fillna((df_eda['BsmtFinType2'].mode()[0])) 
df_eda['BsmtFinType2'].isnull().sum()

plt.hist(df_eda['BsmtFinType2'])

dfDummies_BsmtFinType2= pd.get_dummies(df_eda['BsmtFinType2'], prefix = 'BsmtFinType2') #one hat encoding

del df_eda['BsmtFinType2'] #delete original

del dfDummies_BsmtFinType2['BsmtFinType2_ALQ'] #delete dummy

#36
df.columns[36]

df_eda['BsmtFinSF2'].unique()
df_eda['BsmtFinSF2'].describe()
df_eda['BsmtFinSF2'].count()

plt.hist(df_eda['BsmtFinSF2']) #continuous and numerical
sns.kdeplot(df_eda['BsmtFinSF2'])


df_eda['BsmtFinSF1'].isnull().sum() #check for null values #there are null values
df_eda['BsmtFinSF1'] = df_eda['BsmtFinSF1'].fillna((df_eda['BsmtFinSF1'].mode()))  #since it is year fill with mode
df_eda['BsmtFinSF1'].isnull().sum()

plt.boxplot(df_eda['BsmtFinSF1'])

percentiles = df_eda['BsmtFinSF1'].quantile([0.1,0.9]).values #outliers treatment
df_eda['BsmtFinSF1']=df_eda['BsmtFinSF1'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['BsmtFinSF1'])

#37
df.columns[37]

df_eda['BsmtUnfSF'].unique()
df_eda['BsmtUnfSF'].describe()
df_eda['BsmtUnfSF'].count()

plt.hist(df_eda['BsmtUnfSF']) #continuous and numerical
sns.kdeplot(df_eda['BsmtUnfSF'])


df_eda['BsmtUnfSF'].isnull().sum() #check for null values #there are null values
df_eda['BsmtUnfSF'] = df_eda['BsmtUnfSF'].fillna((df_eda['BsmtUnfSF'].median()))  #since it is year fill with mode
df_eda['BsmtUnfSF'].isnull().sum()

plt.boxplot(df_eda['BsmtUnfSF'])

percentiles = df_eda['BsmtUnfSF'].quantile([0.1,0.9]).values #outliers treatment
df_eda['BsmtUnfSF']=df_eda['BsmtUnfSF'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['BsmtUnfSF'])

#38
df.columns[38]

df_eda['TotalBsmtSF'].unique()
df_eda['TotalBsmtSF'].describe()
df_eda['TotalBsmtSF'].count()

plt.hist(df_eda['TotalBsmtSF']) #continuous and numerical
sns.kdeplot(df_eda['TotalBsmtSF'])


df_eda['TotalBsmtSF'].isnull().sum() #check for null values #there are null values
df_eda['TotalBsmtSF'] = df_eda['TotalBsmtSF'].fillna((df_eda['TotalBsmtSF'].median()))  #since it is year fill with mode
df_eda['TotalBsmtSF'].isnull().sum()

plt.boxplot(df_eda['TotalBsmtSF'])

percentiles = df_eda['TotalBsmtSF'].quantile([0.1,0.9]).values #outliers treatment
df_eda['TotalBsmtSF']=df_eda['TotalBsmtSF'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['TotalBsmtSF'])

#39
df.columns[39]

df_eda['Heating'].unique()
df_eda['Heating'].describe()
df_eda['Heating'].count()

plt.hist(df_eda['Heating'])#categorical and non numerical

df_eda['Heating'].isnull().sum()
df_eda['Heating'] = df_eda['Heating'].fillna((df_eda['Heating'].mode())) 
df_eda['Heating'].isnull().sum()


dfDummies_Heating= pd.get_dummies(df_eda['Heating'], prefix = 'Heating') #one hat encoding

del df_eda['Heating'] #delete original

del dfDummies_Heating['Heating_Floor'] #delete dummy

#40
df.columns[40]

df_eda['HeatingQC'].unique()
df_eda['HeatingQC'].describe()
df_eda['HeatingQC'].count()

plt.hist(df_eda['HeatingQC'])#categorical and non numerical

df_eda['HeatingQC'].isnull().sum()
df_eda['HeatingQC'] = df_eda['HeatingQC'].fillna((df_eda['HeatingQC'].mode())) 
df_eda['HeatingQC'].isnull().sum()


dfDummies_HeatingQC= pd.get_dummies(df_eda['HeatingQC'], prefix = 'HeatingQC') #one hat encoding

del df_eda['HeatingQC'] #delete original

del dfDummies_HeatingQC['HeatingQC_Ex'] #delete dummy

#41
df.columns[41]

df_eda['CentralAir'].unique()
df_eda['CentralAir'].describe()
df_eda['CentralAir'].count()

plt.hist(df_eda['CentralAir'])#categorical and non numerical

df_eda['CentralAir'].isnull().sum()
df_eda['CentralAir'] = df_eda['CentralAir'].fillna((df_eda['CentralAir'].mode())) 
df_eda['CentralAir'].isnull().sum()


dfDummies_CentralAir= pd.get_dummies(df_eda['CentralAir'], prefix = 'CentralAir') #one hat encoding

del df_eda['CentralAir'] #delete original

del dfDummies_CentralAir['CentralAir_N'] #delete dummy

#42
df.columns[42]

df_eda['Electrical'].unique()
df_eda['Electrical'].describe()
df_eda['Electrical'].count() #categorical and non numerical

df_eda['Electrical'].isnull().sum()
df_eda['Electrical'] = df_eda['Electrical'].fillna((df_eda['Electrical'].mode()[0])) 
df_eda['Electrical'].isnull().sum()

plt.hist(df_eda['Electrical'])

dfDummies_Electrical= pd.get_dummies(df_eda['Electrical'], prefix = 'Electrical') #one hat encoding

del df_eda['Electrical'] #delete original

del dfDummies_Electrical['Electrical_FuseA'] #delete dummy

#43
df.columns[43]

df_eda['1stFlrSF'].unique()
df_eda['1stFlrSF'].describe()
df_eda['1stFlrSF'].count()

plt.hist(df_eda['1stFlrSF'])#continuous and numerical
sns.kdeplot(df_eda['1stFlrSF'])


df_eda['1stFlrSF'].isnull().sum() #check for null values #there are null values
df_eda['1stFlrSF'] = df_eda['1stFlrSF'].fillna((df_eda['1stFlrSF'].median()))  #since it is year fill with mode
df_eda['1stFlrSF'].isnull().sum()

plt.boxplot(df_eda['1stFlrSF'])

percentiles = df_eda['1stFlrSF'].quantile([0.1,0.9]).values #outliers treatment
df_eda['1stFlrSF']=df_eda['1stFlrSF'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['1stFlrSF'])

#44
df.columns[44]

df_eda['2ndFlrSF'].unique()
df_eda['2ndFlrSF'].describe()
df_eda['2ndFlrSF'].count()

plt.hist(df_eda['2ndFlrSF'])#continuous and numerical
sns.kdeplot(df_eda['2ndFlrSF'])


df_eda['2ndFlrSF'].isnull().sum() #check for null values #there are null values
df_eda['2ndFlrSF'] = df_eda['2ndFlrSF'].fillna((df_eda['2ndFlrSF'].median()))  #since it is year fill with mode
df_eda['2ndFlrSF'].isnull().sum()

plt.boxplot(df_eda['2ndFlrSF'])

percentiles = df_eda['2ndFlrSF'].quantile([0.1,0.9]).values #outliers treatment
df_eda['2ndFlrSF']=df_eda['2ndFlrSF'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['2ndFlrSF'])

#45
df.columns[45]

df_eda['LowQualFinSF'].unique()
df_eda['LowQualFinSF'].describe()
df_eda['LowQualFinSF'].count()

plt.hist(df_eda['LowQualFinSF'])#continuous and numerical
sns.kdeplot(df_eda['LowQualFinSF'])


df_eda['LowQualFinSF'].isnull().sum() #check for null values #there are null values
df_eda['LowQualFinSF'] = df_eda['LowQualFinSF'].fillna((df_eda['LowQualFinSF'].median()))  #since it is year fill with mode
df_eda['LowQualFinSF'].isnull().sum()

plt.boxplot(df_eda['LowQualFinSF'])

percentiles = df_eda['LowQualFinSF'].quantile([0.1,0.9]).values #outliers treatment
df_eda['LowQualFinSF']=df_eda['LowQualFinSF'].clip(percentiles[0],percentiles[1])

sns.boxplot(df_eda['LowQualFinSF'])

del df_eda['LowQualFinSF']

#46
df.columns[46]

df_eda['GrLivArea'].unique()
df_eda['GrLivArea'].describe()
df_eda['GrLivArea'].count()

plt.hist(df_eda['GrLivArea'])#continuous and numerical
sns.kdeplot(df_eda['GrLivArea'])


df_eda['GrLivArea'].isnull().sum() #check for null values #there are null values
df_eda['GrLivArea'] = df_eda['GrLivArea'].fillna((df_eda['GrLivArea'].median()))  #since it is year fill with mode
df_eda['GrLivArea'].isnull().sum()

plt.boxplot(df_eda['GrLivArea'])

percentiles = df_eda['GrLivArea'].quantile([0.1,0.9]).values #outliers treatment
df_eda['GrLivArea']=df_eda['GrLivArea'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['GrLivArea'])


#47
df.columns[47]

df_eda['BsmtFullBath'].unique()
df_eda['BsmtFullBath'].describe()
df_eda['BsmtFullBath'].count()

plt.hist(df_eda['BsmtFullBath']) #categorical and numrical
sns.kdeplot(df_eda['BsmtFullBath'])

df_eda['BsmtFullBath'].isnull().sum()  #no null values
df_eda['BsmtFullBath'] = df_eda['BsmtFullBath'].fillna((df_eda['BsmtFullBath'].mode())) 
df_eda['BsmtFullBath'].isnull().sum()

plt.boxplot(df_eda['BsmtFullBath']) #to see outliers

percentiles = df_eda['BsmtFullBath'].quantile([0.1,0.9]).values #outliers treatment
df_eda['BsmtFullBath']=df_eda['BsmtFullBath'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['BsmtFullBath'])

dfDummies_BsmtFullBath = pd.get_dummies(df_eda['BsmtFullBath'], prefix = 'BsmtFullBath') #one hat encoding

del df_eda['BsmtFullBath'] #del column in original data

del dfDummies_BsmtFullBath['BsmtFullBath_0'] #del one dummy variable

#48
df.columns[48]

df_eda['BsmtHalfBath'].unique()
df_eda['BsmtHalfBath'].describe()
df_eda['BsmtHalfBath'].count()

plt.hist(df_eda['BsmtHalfBath']) #categorical and numrical
sns.kdeplot(df_eda['BsmtHalfBath'])

df_eda['BsmtHalfBath'].isnull().sum()  #no null values
df_eda['BsmtHalfBath'] = df_eda['BsmtHalfBath'].fillna((df_eda['BsmtHalfBath'].mode())) 
df_eda['BsmtHalfBath'].isnull().sum()

plt.boxplot(df_eda['BsmtHalfBath']) #to see outliers

percentiles = df_eda['BsmtHalfBath'].quantile([0.1,0.9]).values #outliers treatment
df_eda['BsmtHalfBath']=df_eda['BsmtHalfBath'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['BsmtHalfBath'])

dfDummies_BsmtHalfBath= pd.get_dummies(df_eda['BsmtHalfBath'], prefix = 'BsmtHalfBath') #one hat encoding

del df_eda['BsmtHalfBath'] #del column in original data

del dfDummies_BsmtHalfBath['BsmtHalfBath_0'] #del one dummy variable

#49
df.columns[49]

df_eda['FullBath'].unique()
df_eda['FullBath'].describe()
df_eda['FullBath'].count()

plt.hist(df_eda['FullBath'])#categorical and numrical
sns.kdeplot(df_eda['FullBath'])

df_eda['FullBath'].isnull().sum()  #no null values
df_eda['FullBath'] = df_eda['FullBath'].fillna((df_eda['FullBath'].mode())) 
df_eda['FullBath'].isnull().sum()

plt.boxplot(df_eda['FullBath']) #to see outliers

percentiles = df_eda['FullBath'].quantile([0.1,0.9]).values #outliers treatment
df_eda['FullBath']=df_eda['FullBath'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['FullBath'])

dfDummies_FullBath= pd.get_dummies(df_eda['FullBath'], prefix = 'FullBath') #one hat encoding

del df_eda['FullBath'] #del column in original data

del dfDummies_FullBath['FullBath_1'] #del one dummy variable

#50
df.columns[50]

df_eda['HalfBath'].unique()
df_eda['HalfBath'].describe()
df_eda['HalfBath'].count()

plt.hist(df_eda['HalfBath'])#categorical and numrical
sns.kdeplot(df_eda['HalfBath'])

df_eda['HalfBath'].isnull().sum()  #no null values
df_eda['HalfBath'] = df_eda['HalfBath'].fillna((df_eda['HalfBath'].mode())) 
df_eda['HalfBath'].isnull().sum()

plt.boxplot(df_eda['HalfBath']) #to see outliers

percentiles = df_eda['HalfBath'].quantile([0.1,0.9]).values #outliers treatment
df_eda['HalfBath']=df_eda['HalfBath'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['HalfBath'])

dfDummies_HalfBath= pd.get_dummies(df_eda['HalfBath'], prefix = 'HalfBath') #one hat encoding

del df_eda['HalfBath'] #del column in original data

del dfDummies_HalfBath['HalfBath_0'] #del one dummy variable


#51
df.columns[51]

df_eda['BedroomAbvGr'].unique()
df_eda['BedroomAbvGr'].describe()
df_eda['BedroomAbvGr'].count()

plt.hist(df_eda['BedroomAbvGr']) #categorical and numrical
sns.kdeplot(df_eda['BedroomAbvGr'])

df_eda['BedroomAbvGr'].isnull().sum()  #no null values
df_eda['BedroomAbvGr'] = df_eda['BedroomAbvGr'].fillna((df_eda['BedroomAbvGr'].mode())) 
df_eda['BedroomAbvGr'].isnull().sum()

plt.boxplot(df_eda['BedroomAbvGr']) #to see outliers

percentiles = df_eda['BedroomAbvGr'].quantile([0.1,0.9]).values #outliers treatment
df_eda['BedroomAbvGr']=df_eda['BedroomAbvGr'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['BedroomAbvGr'])

dfDummies_BedroomAbvGr= pd.get_dummies(df_eda['BedroomAbvGr'], prefix = 'BedroomAbvGr') #one hat encoding

del df_eda['BedroomAbvGr'] #del column in original data

del dfDummies_BedroomAbvGr['BedroomAbvGr_2'] #del one dummy variable

#52
df.columns[52]

df_eda['KitchenAbvGr'].unique()
df_eda['KitchenAbvGr'].describe()
df_eda['KitchenAbvGr'].count()

plt.hist(df_eda['KitchenAbvGr'])#categorical and numrical
sns.kdeplot(df_eda['KitchenAbvGr'])

df_eda['KitchenAbvGr'].isnull().sum()  #no null values
df_eda['KitchenAbvGr'] = df_eda['KitchenAbvGr'].fillna((df_eda['KitchenAbvGr'].mode())) 
df_eda['KitchenAbvGr'].isnull().sum()

plt.boxplot(df_eda['KitchenAbvGr']) #to see outliers

percentiles = df_eda['KitchenAbvGr'].quantile([0.1,0.9]).values #outliers treatment
df_eda['KitchenAbvGr']=df_eda['KitchenAbvGr'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['KitchenAbvGr'])

dfDummies_KitchenAbvGr= pd.get_dummies(df_eda['KitchenAbvGr'], prefix = 'KitchenAbvGr') #one hat encoding

del df_eda['KitchenAbvGr'] #del column in original data

del dfDummies_KitchenAbvGr['KitchenAbvGr_1'] #del one dummy variable


#53
df.columns[53]

df_eda['KitchenQual'].unique()
df_eda['KitchenQual'].describe()
df_eda['KitchenQual'].count()

plt.hist(df_eda['KitchenQual'])#categorical and non numerical

df_eda['KitchenQual'].isnull().sum()
df_eda['KitchenQual'] = df_eda['KitchenQual'].fillna((df_eda['KitchenQual'].mode())) 
df_eda['KitchenQual'].isnull().sum()


dfDummies_KitchenQual= pd.get_dummies(df_eda['KitchenQual'], prefix = 'KitchenQual') #one hat encoding

del df_eda['KitchenQual'] #delete original

del dfDummies_KitchenQual['KitchenQual_Ex'] #delete dummy

#54
df.columns[54]

df_eda['TotRmsAbvGrd'].unique()
df_eda['TotRmsAbvGrd'].describe()
df_eda['TotRmsAbvGrd'].count()

plt.hist(df_eda['TotRmsAbvGrd'])#categorical and numrical
sns.kdeplot(df_eda['TotRmsAbvGrd'])

df_eda['TotRmsAbvGrd'].isnull().sum()  #no null values
df_eda['TotRmsAbvGrd'] = df_eda['TotRmsAbvGrd'].fillna((df_eda['TotRmsAbvGrd'].mode())) 
df_eda['TotRmsAbvGrd'].isnull().sum()

plt.boxplot(df_eda['TotRmsAbvGrd']) #to see outliers

percentiles = df_eda['TotRmsAbvGrd'].quantile([0.1,0.9]).values #outliers treatment
df_eda['TotRmsAbvGrd']=df_eda['TotRmsAbvGrd'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['TotRmsAbvGrd'])

dfDummies_TotRmsAbvGrd= pd.get_dummies(df_eda['TotRmsAbvGrd'], prefix = 'TotRmsAbvGrd') #one hat encoding

del df_eda['TotRmsAbvGrd'] #del column in original data

del dfDummies_TotRmsAbvGrd['TotRmsAbvGrd_5'] #del one dummy variable

#55 Functional
df.columns[55]

df_eda['Functional'].unique()
df_eda['Functional'].describe()
df_eda['Functional'].count()

plt.hist(df_eda['Functional'])#categorical and non numerical

df_eda['Functional'].isnull().sum()
df_eda['Functional'] = df_eda['Functional'].fillna((df_eda['Functional'].mode())) 
df_eda['Functional'].isnull().sum()


dfDummies_Functional= pd.get_dummies(df_eda['Functional'], prefix = 'Functional') #one hat encoding

del df_eda['Functional'] #delete original

del dfDummies_Functional['Functional_Maj1'] #delete dummy

#56 Fireplaces
df.columns[56]

df_eda['Fireplaces'].unique()
df_eda['Fireplaces'].describe()
df_eda['Fireplaces'].count()

plt.hist(df_eda['Fireplaces'])#categorical and numrical
sns.kdeplot(df_eda['Fireplaces'])

df_eda['Fireplaces'].isnull().sum()  #no null values
df_eda['Fireplaces'] = df_eda['Fireplaces'].fillna((df_eda['Fireplaces'].mode())) 
df_eda['Fireplaces'].isnull().sum()

plt.boxplot(df_eda['Fireplaces']) #to see outliers

percentiles = df_eda['Fireplaces'].quantile([0.1,0.9]).values #outliers treatment
df_eda['Fireplaces']=df_eda['Fireplaces'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['Fireplaces'])

dfDummies_Fireplaces= pd.get_dummies(df_eda['Fireplaces'], prefix = 'Fireplaces') #one hat encoding

del df_eda['Fireplaces'] #del column in original data

del dfDummies_Fireplaces['Fireplaces_0'] #del one dummy variable

#57 FireplaceQu
df.columns[57]

df_eda['FireplaceQu'].unique()
df_eda['FireplaceQu'].describe()
df_eda['FireplaceQu'].count()

df_eda['FireplaceQu'].isnull().sum()
df_eda['FireplaceQu'] = df_eda['FireplaceQu'].fillna((df_eda['FireplaceQu'].mode()[0])) 
df_eda['FireplaceQu'].isnull().sum()

plt.hist(df_eda['FireplaceQu'])

dfDummies_FireplaceQu= pd.get_dummies(df_eda['FireplaceQu'], prefix = 'FireplaceQu') #one hat encoding

del df_eda['FireplaceQu'] #delete original

del dfDummies_FireplaceQu['FireplaceQu_Ex'] #delete dummy

#58 GarageType
df.columns[58]

df_eda['GarageType'].unique()
df_eda['GarageType'].describe()
df_eda['GarageType'].count()   #categorical and non numerical

df_eda['GarageType'].isnull().sum()
df_eda['GarageType'] = df_eda['GarageType'].fillna((df_eda['GarageType'].mode()[0])) 
df_eda['GarageType'].isnull().sum()

plt.hist(df_eda['GarageType'])

dfDummies_GarageType= pd.get_dummies(df_eda['GarageType'], prefix = 'GarageType') #one hat encoding

del df_eda['GarageType'] #delete original

del dfDummies_GarageType['GarageType_2Types'] #delete dummy

#59 GarageYrBlt
df.columns[59]

df_eda['GarageYrBlt'].unique()
df_eda['GarageYrBlt'].describe()
df_eda['GarageYrBlt'].count()

plt.hist(df_eda['GarageYrBlt']) #continuous and numerical
sns.kdeplot(df_eda['GarageYrBlt'])


df_eda['GarageYrBlt'].isnull().sum() #check for null values #there are null values
df_eda['GarageYrBlt'] = df_eda['GarageYrBlt'].fillna((df_eda['GarageYrBlt'].mean())) 
df_eda['GarageYrBlt'].isnull().sum()

plt.boxplot(df_eda['GarageYrBlt'])

percentiles = df_eda['GarageYrBlt'].quantile([0.1,0.9]).values #outliers treatment
df_eda['GarageYrBlt']=df_eda['GarageYrBlt'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['GarageYrBlt'])

#60 GarageFinish
df.columns[60]

df_eda['GarageFinish'].unique()
df_eda['GarageFinish'].describe()
df_eda['GarageFinish'].count()

df_eda['GarageFinish'].isnull().sum() #categorical and non numerical
df_eda['GarageFinish'] = df_eda['GarageFinish'].fillna((df_eda['GarageFinish'].mode()[0])) 
df_eda['GarageFinish'].isnull().sum()

plt.hist(df_eda['GarageFinish'])

dfDummies_GarageFinish= pd.get_dummies(df_eda['GarageFinish'], prefix = 'GarageFinish') #one hat encoding

del df_eda['GarageFinish'] #delete original

del dfDummies_GarageFinish['GarageFinish_Fin']


#61
df.columns[61]

df_eda['GarageCars'].unique()
df_eda['GarageCars'].describe()
df_eda['GarageCars'].count()

plt.hist(df_eda['GarageCars']) #categorical and numrical
sns.kdeplot(df_eda['GarageCars'])

df_eda['GarageCars'].isnull().sum()  #no null values
df_eda['GarageCars'] = df_eda['GarageCars'].fillna((df_eda['GarageCars'].mode())) 
df_eda['GarageCars'].isnull().sum()

plt.boxplot(df_eda['GarageCars']) #to see outliers

percentiles = df_eda['GarageCars'].quantile([0.1,0.9]).values #outliers treatment
df_eda['GarageCars']=df_eda['GarageCars'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['GarageCars'])

dfDummies_GarageCars= pd.get_dummies(df_eda['GarageCars'], prefix = 'GarageCars') #one hat encoding

del df_eda['GarageCars'] #del column in original data

del dfDummies_GarageCars['GarageCars_1'] #del one dummy variable


#62
df.columns[62]

df_eda['GarageArea'].unique()
df_eda['GarageArea'].describe()
df_eda['GarageArea'].count()

plt.hist(df_eda['GarageArea'])  #continuous and numerical
sns.kdeplot(df_eda['GarageArea'])


df_eda['GarageArea'].isnull().sum() #check for null values #there are null values
df_eda['GarageArea'] = df_eda['GarageArea'].fillna((df_eda['GarageArea'].mean())) 
df_eda['GarageArea'].isnull().sum()

plt.boxplot(df_eda['GarageArea'])

percentiles = df_eda['GarageArea'].quantile([0.1,0.9]).values #outliers treatment
df_eda['GarageArea']=df_eda['GarageArea'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['GarageArea'])

#63 GarageQual
df.columns[63]

df_eda['GarageQual'].describe()
df_eda['GarageQual'].count()
df_eda['GarageQual'].unique()

df_eda['GarageQual'].isnull().sum() #categorical and non numerical
df_eda['GarageQual'] = df_eda['GarageQual'].fillna((df_eda['GarageQual'].mode()[0])) 
df_eda['GarageQual'].isnull().sum()

plt.hist(df_eda['GarageQual'])

dfDummies_GarageQual= pd.get_dummies(df_eda['GarageQual'], prefix = 'GarageQual') #one hat encoding

del df_eda['GarageQual'] #delete original

del dfDummies_GarageQual['GarageQual_Ex']

#64 GarageCond
df.columns[64]

df_eda['GarageCond'].describe()
df_eda['GarageCond'].count()
df_eda['GarageCond'].unique()

df_eda['GarageCond'].isnull().sum() #categorical and non numerical
df_eda['GarageCond'] = df_eda['GarageCond'].fillna((df_eda['GarageCond'].mode()[0])) 
df_eda['GarageCond'].isnull().sum()

plt.hist(df_eda['GarageCond'])

dfDummies_GarageCond= pd.get_dummies(df_eda['GarageCond'], prefix = 'GarageCond') #one hat encoding

del df_eda['GarageCond'] #delete original

del dfDummies_GarageCond['GarageCond_Ex']

#65 PavedDrive
df.columns[65]

df_eda['PavedDrive'].describe()
df_eda['PavedDrive'].count()
df_eda['PavedDrive'].unique()


plt.hist(df_eda['PavedDrive']) #categorical and non numerical

df_eda['PavedDrive'].isnull().sum()
df_eda['PavedDrive'] = df_eda['PavedDrive'].fillna((df_eda['PavedDrive'].mode())) 
df_eda['PavedDrive'].isnull().sum()


dfDummies_PavedDrive= pd.get_dummies(df_eda['PavedDrive'], prefix = 'PavedDrive') #one hat encoding

del df_eda['PavedDrive'] #delete original

del dfDummies_PavedDrive['PavedDrive_N'] #delete dummy

#66 WoodDeckSF
df.columns[66]

df_eda['WoodDeckSF'].describe()
df_eda['WoodDeckSF'].count()
df_eda['WoodDeckSF'].unique()


plt.hist(df_eda['WoodDeckSF'])#continuous and numerical
sns.kdeplot(df_eda['WoodDeckSF'])


df_eda['WoodDeckSF'].isnull().sum() #check for null values #there are null values
df_eda['WoodDeckSF'] = df_eda['WoodDeckSF'].fillna((df_eda['WoodDeckSF'].mean())) 
df_eda['WoodDeckSF'].isnull().sum()

plt.boxplot(df_eda['WoodDeckSF'])

percentiles = df_eda['WoodDeckSF'].quantile([0.1,0.9]).values #outliers treatment
df_eda['WoodDeckSF']=df_eda['WoodDeckSF'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['WoodDeckSF'])


#67 OpenPorchSF
df.columns[67]

df_eda['OpenPorchSF'].describe()
df_eda['OpenPorchSF'].count()
df_eda['OpenPorchSF'].unique()


plt.hist(df_eda['OpenPorchSF'])#continuous and numerical
sns.kdeplot(df_eda['OpenPorchSF'])


df_eda['OpenPorchSF'].isnull().sum() #check for null values #there are null values
df_eda['OpenPorchSF'] = df_eda['OpenPorchSF'].fillna((df_eda['OpenPorchSF'].mean())) 
df_eda['OpenPorchSF'].isnull().sum()

plt.boxplot(df_eda['OpenPorchSF'])

percentiles = df_eda['OpenPorchSF'].quantile([0.1,0.9]).values #outliers treatment
df_eda['OpenPorchSF']=df_eda['OpenPorchSF'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['OpenPorchSF'])

#68 EnclosedPorch
df.columns[68]

df_eda['EnclosedPorch'].describe()  #maximum are zero
df_eda['EnclosedPorch'].count()
df_eda['EnclosedPorch'].unique()

del df_eda['EnclosedPorch']

#69 3SsnPorch
df.columns[69]

df_eda['3SsnPorch'].describe() #maximum are zero
df_eda['3SsnPorch'].count()
df_eda['3SsnPorch'].unique()

del df_eda['3SsnPorch'] 

#70 ScreenPorch
df.columns[70]

df_eda['ScreenPorch'].describe() #maximum are zero
df_eda['ScreenPorch'].count()
df_eda['ScreenPorch'].unique()

del df_eda['ScreenPorch']

#71 PoolArea
df.columns[71]

df_eda['PoolArea'].describe() #maximum are zero
df_eda['PoolArea'].count()
df_eda['PoolArea'].unique()

del df_eda['PoolArea']

#72 PoolQC
df.columns[72]

df_eda['PoolQC'].describe() 
df_eda['PoolQC'].count()    #maximum nan valuex in the data column
df_eda['PoolQC'].unique()

del df_eda['PoolQC'] 


#73 Fence
df.columns[73]

df_eda['Fence'].describe() 
df_eda['Fence'].count() #maximum nan valuex in the data column
df_eda['Fence'].unique()

del df_eda['Fence'] 

#74 MiscFeature
df.columns[74]

df_eda['MiscFeature'].describe() 
df_eda['MiscFeature'].count() #maximum nan valuex in the data column
df_eda['MiscFeature'].unique()

del df_eda['MiscFeature'] 

#75 MiscVal
df.columns[75]

df_eda['MiscVal'].describe() 
df_eda['MiscVal'].count() #maximum 0 value in the data column
df_eda['MiscVal'].unique()

del df_eda['MiscVal']

#76 MoSold
df.columns[76]

df_eda['MoSold'].describe() 
df_eda['MoSold'].count()
df_eda['MoSold'].unique()

plt.hist(df_eda['MoSold'])#categorical and numrical
sns.kdeplot(df_eda['MoSold'])

df_eda['MoSold'].isnull().sum()  #no null values
df_eda['MoSold'] = df_eda['MoSold'].fillna((df_eda['MoSold'].mode())) 
df_eda['MoSold'].isnull().sum()

plt.boxplot(df_eda['MoSold']) #to see outliers

percentiles = df_eda['MoSold'].quantile([0.1,0.9]).values #outliers treatment
df_eda['MoSold']=df_eda['MoSold'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['MoSold'])

dfDummies_MoSold= pd.get_dummies(df_eda['MoSold'], prefix = 'MoSold') #one hat encoding

del df_eda['MoSold'] #del column in original data

del dfDummies_MoSold['MoSold_3'] #del one dummy variable


#77 YrSold

df.columns[77]

df_eda['YrSold'].describe() 
df_eda['YrSold'].count()
df_eda['YrSold'].unique()

plt.hist(df_eda['YrSold']) #categorical and numrical
sns.kdeplot(df_eda['YrSold'])

df_eda['YrSold'].isnull().sum()  #no null values
df_eda['YrSold'] = df_eda['YrSold'].fillna((df_eda['YrSold'].mode())) 
df_eda['YrSold'].isnull().sum()

plt.boxplot(df_eda['YrSold']) #to see outliers

percentiles = df_eda['YrSold'].quantile([0.1,0.9]).values #outliers treatment
df_eda['YrSold']=df_eda['YrSold'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['YrSold'])

dfDummies_YrSold= pd.get_dummies(df_eda['YrSold'], prefix = 'YrSold') #one hat encoding

del df_eda['YrSold'] #del column in original data

del dfDummies_YrSold['YrSold_2006'] #del one dummy variable

#78 SaleType
df.columns[78]

df_eda['SaleType'].describe() 
df_eda['SaleType'].count()
df_eda['SaleType'].unique()

plt.hist(df_eda['SaleType'])#categorical and non numerical

df_eda['SaleType'].isnull().sum()
df_eda['SaleType'] = df_eda['SaleType'].fillna((df_eda['SaleType'].mode())) 
df_eda['SaleType'].isnull().sum()


dfDummies_SaleType= pd.get_dummies(df_eda['SaleType'], prefix = 'SaleType') #one hat encoding

del df_eda['SaleType'] #delete original

del dfDummies_SaleType['SaleType_COD'] #delete dummy

#79 SaleCondition
df.columns[79]

df_eda['SaleCondition'].describe() 
df_eda['SaleCondition'].count()
df_eda['SaleCondition'].unique()

plt.hist(df_eda['SaleCondition'])#categorical and non numerical

df_eda['SaleCondition'].isnull().sum()
df_eda['SaleCondition'] = df_eda['SaleCondition'].fillna((df_eda['SaleCondition'].mode())) 
df_eda['SaleCondition'].isnull().sum()


dfDummies_SaleCondition= pd.get_dummies(df_eda['SaleCondition'], prefix = 'SaleCondition') #one hat encoding

del df_eda['SaleCondition'] #delete original

del dfDummies_SaleCondition['SaleCondition_Abnorml'] #delete dummy

#80 SalePrice
df.columns[80]

df_eda['SalePrice'].describe() 
df_eda['SalePrice'].count()
df_eda['SalePrice'].unique()

plt.hist(df_eda['SalePrice']) #continuous and numerical
sns.kdeplot(df_eda['SalePrice'])


df_eda['SalePrice'].isnull().sum() #check for null values #there are null values
df_eda['SalePrice'] = df_eda['SalePrice'].fillna((df_eda['SalePrice'].median())) 
df_eda['SalePrice'].isnull().sum()

plt.boxplot(df_eda['SalePrice'])

percentiles = df_eda['SalePrice'].quantile([0.1,0.9]).values #outliers treatment
df_eda['SalePrice']=df_eda['SalePrice'].clip(percentiles[0],percentiles[1])

plt.boxplot(df_eda['SalePrice'])


data_conc = pd.concat([df_eda,
    dfDummies_MSSubClass,
    dfDummies_MSZoning,
    dfDummies_Street,
    dfDummies_LotShape,
    dfDummies_LandContour,
    dfDummies_Utilities,
    dfDummies_LotConfig,
    dfDummies_LandSlope,
    dfDummies_Neighborhood,
    dfDummies_Condition1,
    dfDummies_Condition2,
    dfDummies_BldgType,
    dfDummies_HouseStyle,
    dfDummies_OverallQual,
    dfDummies_OverallCond,
    dfDummies_RoofStyle,
    dfDummies_RoofMatl,
    dfDummies_Exterior1st,
    dfDummies_Exterior2nd,
    dfDummies_ExterQual,
    dfDummies_ExterCond,
    dfDummies_Heating,
    dfDummies_HeatingQC,
    dfDummies_CentralAir,
    dfDummies_BsmtFullBath,
    dfDummies_BsmtHalfBath,
    dfDummies_FullBath,
    dfDummies_HalfBath,
    dfDummies_BedroomAbvGr,
    dfDummies_KitchenAbvGr,
    dfDummies_KitchenQual,
    dfDummies_TotRmsAbvGrd,
    dfDummies_TotRmsAbvGrd,
    dfDummies_Functional,
    dfDummies_Fireplaces,
    dfDummies_GarageCars,
    dfDummies_PavedDrive,
    dfDummies_MoSold,
    dfDummies_YrSold,
    dfDummies_SaleType,
    dfDummies_SaleCondition,
    dfDummies_BsmtQual,
    dfDummies_MasVnrType,
    dfDummies_BsmtCond,
    dfDummies_BsmtExposure,
    dfDummies_BsmtFinType1,
    dfDummies_BsmtFinType2,
    dfDummies_Electrical,
    dfDummies_FireplaceQu,
    dfDummies_GarageType,
    dfDummies_GarageFinish,
    dfDummies_GarageQual,
    dfDummies_GarageCond],axis=1)


data_conc


data_conc.to_csv("C:/Users/RANVI/Desktop/Silfra assignment- Rajkamal/3.Cleaned output/cleaned_train_data.csv',index=False)

cleaned_train_data=pd.read_csv("C:/Users/RANVI/Desktop/Silfra assignment- Rajkamal/3.Cleaned output/cleaned_train_data.csv')

