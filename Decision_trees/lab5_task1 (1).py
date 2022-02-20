# -*- coding: utf-8 -*-
"""Lab5- Task1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Spb1bqoxtNNUDB-Y5HTv83mOsAP3MXF1
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score, roc_auc_score,roc_curve
from sklearn.preprocessing import LabelEncoder

from google.colab import drive
drive.mount("/content/gdrive")

from google.colab import files
uploaded = files.upload()

df=pd.read_csv('drugs.csv')
df.head()

"""1.1 Target is drugs and features are rest of columns provided

1.2
"""

df.loc[:, df.isnull().any()].columns

"""Hence no null values present"""

from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
oe=OrdinalEncoder()
encoded_df=oe.fit_transform(df[['BP','Cholesterol']])
df1=pd.DataFrame(encoded_df,columns=['BP','Cholesterol'])
df['BP']=df1['BP']
df['Cholesterol']=df1['Cholesterol']
df['Drug'].unique()

ohe=OneHotEncoder(sparse=False)
encoded_ohe=ohe.fit_transform(df[['Sex','Drug']])

df2=pd.DataFrame(encoded_ohe,columns=['Male','Female','DrugY', 'drugC', 'drugX', 'drugA', 'drugB'])
final=pd.concat([df,df2],axis=1)
new_df=final.drop(['Sex','Drug'],axis=1)
new_df

X=new_df.iloc[:,0:6]
Y=new_df.iloc[:,6:11]



"""**When ratio is 70:30**"""

X_train,X_test,y_train,y_test=train_test_split(X,Y,shuffle=True,test_size=0.3)



def decision_tree(X_train,y_train,X_test,y_test,criterion):
  tiree=DecisionTreeClassifier(criterion=criterion,random_state=55)
  tiree.fit(X_train,y_train)
  pred = tiree.predict(X_train)
  print('Model Accuracy | ',accuracy_score(pred,y_train))
  pred_ytest = tiree.predict(X_test)
  return tiree,pred_ytest
tiree,pred_ytest=decision_tree(X_train,y_train,X_test,y_test,'gini')
tiree

tiree.tree_.max_depth



conf_mat = confusion_matrix(y_test.values.argmax(axis=1), pred_ytest.argmax(axis=1))
conf_mat

pred = tiree.predict(X_train)
print('Model Accuracy | ',accuracy_score(pred,y_train))
pred_ytest = tiree.predict(X_test)
print('Test Accuracy | ',accuracy_score(pred_ytest,y_test))

print('Accuracy | ',accuracy_score(y_test,pred_ytest))
print('Precision | ',precision_score(y_test,pred_ytest,average='micro'))
print('Recall |' ,recall_score(y_test,pred_ytest,average='micro'))
print('F1-score | ',f1_score(y_test,pred_ytest,average='micro'))

import graphviz
dot_tree = tree.export_graphviz(tiree,filled=True,rounded=True)
graph = graphviz.Source(dot_tree, format="png") 
graph



graph.render("decision_tree_graphivz")

"""**When ratio is 80:20**"""

X_train2,X_test2,y_train2,y_test2=train_test_split(X,Y,shuffle=True,test_size=0.2)

decision,pred=decision_tree(X_train2,y_train2,X_test2,y_test2,'entropy')

decision.tree_.max_depth

conf_mat = confusion_matrix(y_test2.values.argmax(axis=1), pred.argmax(axis=1))
conf_mat

pred_train = decision.predict(X_train2)
print('Model Accuracy | ',accuracy_score(pred_train,y_train2))
pred = decision.predict(X_test2)
print('Test Accuracy | ',accuracy_score(pred,y_test2))

import graphviz
dot_tree = tree.export_graphviz(decision,filled=True,rounded=True)
graph = graphviz.Source(dot_tree, format="png") 
graph

print('Accuracy | ',accuracy_score(y_test2,pred))
print('Precision | ',precision_score(y_test2,pred,average='micro'))
print('Recall |' ,recall_score(y_test2,pred,average='micro'))
print('F1-score | ',f1_score(y_test2,pred,average='micro'))

"""**When ratio is 90:10**

"""

X_train3,X_test3,y_train3,y_test3=train_test_split(X,Y,shuffle=True,test_size=0.1)

decision2,pred2=decision_tree(X_train3,y_train3,X_test3,y_test3,'gini')

conf_mat = confusion_matrix(y_test3.values.argmax(axis=1), pred2.argmax(axis=1))
conf_mat

pred_train = decision2.predict(X_train3)
print('Model Accuracy | ',accuracy_score(pred_train,y_train3))
pred3 = decision2.predict(X_test3)
print('Test Accuracy | ',accuracy_score(pred3,y_test3))

print('Accuracy | ',accuracy_score(y_test3,pred2))
print('Precision | ',precision_score(y_test3,pred2,average='micro'))
print('Recall |' ,recall_score(y_test3,pred2,average='micro'))
print('F1-score | ',f1_score(y_test3,pred2,average='micro'))