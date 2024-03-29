# -*- coding: utf-8 -*-
"""Lab10-B20ME027.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CeQLcUFQN72NZSo0uIZ9SmrJdMle20T2
"""

from sklearn.neural_network import MLPClassifier,MLPRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
import matplotlib.pyplot as plt

df=pd.read_csv('iris.data')
df.columns=['feature1','feature2','feature3','feature4','labels']
X=df.iloc[:,0:4]
y=df.iloc[:,4]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

MLP=MLPClassifier(hidden_layer_sizes=2,max_iter=50,learning_rate_init=0.1)
MLP.fit(X_train,y_train)
y_pred=MLP.predict(X_test)
MLP.score(X_test,y_test)

MLP=MLPClassifier(hidden_layer_sizes=2,max_iter=50,learning_rate_init=0.01)
MLP.fit(X_train,y_train)
y_pred1=MLP.predict(X_test)
MLP.score(X_test,y_test)

MLP=MLPClassifier(hidden_layer_sizes=2,max_iter=50,learning_rate_init=0.0001)
MLP.fit(X_train,y_train)
y_pred2=MLP.predict(X_test)
MLP.score(X_test,y_test)

MLP=MLPClassifier(hidden_layer_sizes=2,max_iter=100,learning_rate_init=0.1)
MLP.fit(X_train,y_train)
MLP.predict(X_test)
loss_values=MLP.loss_curve_
l=[]
for i in range(len(loss_values)+1):
    if i%10==0:
        l.append(loss_values[i-1])
l

l2=[10,20,30,40,50,60,70,80,90,100,101]
plt.plot(l2,l)

data=pd.read_csv('train.csv')
X_reg=data.iloc[:,0:80]
y_reg=data.iloc[:,80]

X=X_reg.drop(['Alley','Fence','MiscFeature','PoolArea','Street'],axis=1)

from sklearn.impute import SimpleImputer
c1=X.loc[:, X.isnull().any()].columns
imp=SimpleImputer(strategy='mean')
new=imp.fit_transform(X[['LotFrontage','MasVnrArea','GarageYrBlt']])
X['LotFrontage']=new[:,0]
X['MasVnrArea']=new[:,1]
X['GarageYrBlt']=new[:,2]
print(len(c1))

num={'MasVnrType':{'BrkFace':1, 'None':2, 'Stone':3, 'BrkCmn':4},'BsmtQual':{'Gd':1 ,'TA':2, 'Ex':3, 'Fa':4},'BsmtCond':{'TA':3, 'Gd':2, 'Fa':1, 'Po':0},'BsmtExposure': {'No':1, 'Gd':2, 'Mn':3, 'Av':4},'BsmtFinType1':{'GLQ':1, 'ALQ':2, 'Unf':3, 'Rec':4, 'BLQ':5, 'LwQ':6},'BsmtFinType2': {'Unf':1, 'BLQ':2, 'ALQ':3, 'Rec':4, 'LwQ':5, 'GLQ':5},'Electrical': {'SBrkr':1, 'FuseF':2, 'FuseA':3, 'FuseP':4, 'Mix':5},'FireplaceQu': {'TA':1, 'Gd':2, 'Fa':3, 'Ex':4, 'Po':5},'GarageType': {'Attchd':1, 'Detchd':2, 'BuiltIn':3, 'CarPort':4, 'Basment':5, '2Types':6},'GarageFinish': {'RFn':1, 'Unf':2, 'Fin':3},'GarageQual':{'TA':1, 'Fa':2, 'Gd':3, 'Ex':4, 'Po':5},'GarageCond': {'TA':1, 'Fa':2, 'Gd':3, 'Po':5, 'Ex':4},'PoolQC': {'Ex':1, 'Fa':2, 'Gd':3}}

X.replace(num, inplace = True)
imp1=SimpleImputer(strategy='median')
new1=imp1.fit_transform(X[['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType','GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC']])
X[c1[0]]=new1[:,0]
X[c1[1]]=new1[:,1]
X[c1[2]]=new1[:,2]
X[c1[3]]=new1[:,3]
X[c1[4]]=new1[:,4]
X[c1[5]]=new1[:,5]
X[c1[6]]=new1[:,6]
X[c1[7]]=new1[:,7]
X[c1[8]]=new1[:,8]
X[c1[9]]=new1[:,9]
X[c1[10]]=new1[:,10]
X[c1[11]]=new1[:,11]

c1=X.loc[:, X.isnull().any()].columns
l=[]
for i in X.columns:
    if X[i].dtypes=='object':
        l.append(i)
oe=OrdinalEncoder()
transformed=oe.fit_transform(X[l])
datafr=pd.DataFrame(transformed,columns=l)
for i in datafr.columns:
    X[i]=datafr[i]
X_train1,X_test1,y_train1,y_test1=train_test_split(X,y_reg,test_size=0.3)

from sklearn.neural_network import MLPRegressor
regr=MLPRegressor(activation='identity',hidden_layer_sizes=300,learning_rate_init=0.001)
regr.fit(X_train1,y_train1)
y_pred=regr.predict(X_test1)
regr.score(X_test1,y_test1)

y_pred[0:4]