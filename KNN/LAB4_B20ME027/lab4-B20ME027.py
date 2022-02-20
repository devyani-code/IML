#!/usr/bin/env python
# coding: utf-8

# In[360]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# In[361]:


df=pd.read_csv('diabetes.csv')


# In[362]:


X=df.drop(['Outcome'], axis=1)
Y=df['Outcome']
X_1,X_test,y_1,y_test=train_test_split(X,Y,test_size=0.15,shuffle=False)
X_train,X_valid,y_train,y_valid=train_test_split(X_1,y_1,test_size=0.1275)


# In[363]:


X_test


# In[364]:


X_train=np.matrix(X_train)
X_valid=np.matrix(X_valid)


# In[365]:


X_test=np.matrix(X_test)
X_test.shape
y_train=np.matrix(y_train)


# In[366]:


y_train=np.array(y_train)
y_train.flatten()
p=y_train.tolist()


# In[367]:


def KNN(X_test,X_train,K,y_train):
 final=[]
 for value in range(len(X_test)):
  distance=[]
  close=[]
  for j in range(len(X_train)):
    eucledian_distance=np.sqrt(np.sum(np.square(X_train[j]-X_test[value])))
    distance.append(eucledian_distance)

  new=sorted(distance)
  for i in range(len(new)):
    if i<K:
        close.append(new[i])
  index=[]
  for i in close:
    for j in range(len(distance)):
        if i==distance[j]:
            index.append(j)
  out=[]
  for i in index:
    for j in range(len(p[0])):
        if i==j:
            out.append(p[0][j])
  if out.count(0)>out.count(1):
      final.append(0)
  elif out.count(0)<out.count(1):
      final.append(1)
  else:
    n=close.index(min(close))
    final.append(out[n])
 return final 
final=KNN(X_test,X_train,K,y_train)


# In[368]:


er=[]
k=[j for j in range(5,10)]
for i in range(5,10):
    y_pred=KNN(X_test,X_train,i,y_train)
    er.append(np.mean(y_pred!=y_test))
er


# In[369]:


plt.plot(k,er)
plt.title('Error Rate vs K')
plt.xlabel('values of K')
plt.ylabel('error rate')


# In[370]:


for i in k:
  y_pred=KNN(X_valid,X_train,i,y_train)
  print(confusion_matrix(y_valid,y_pred))
  print(classification_report(y_valid,y_pred))


# In[356]:


y_pred=KNN(X_test,X_train,8,y_train)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[371]:


X1,X_test,y1,y_test=train_test_split(X,Y,test_size=0.15)
X2,X_valid,y2,y_valid=train_test_split(X1,y1,test_size=0.15)
knn=KNeighborsClassifier(n_neighbors=8)
model=knn.fit(X2,y2)


# In[372]:


pred=knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[ ]:




