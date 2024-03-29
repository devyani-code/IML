# -*- coding: utf-8 -*-
"""iml lab7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I6z3IRtp67sKQgB8HNcHPpc9_brqmCYl
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import linalg as LA
from matplotlib import pyplot as plt

dataset=pd.read_csv('mnist_train.csv')
df=dataset.iloc[0:5000,:]
df=df.drop(columns=['label'])
df

X=np.matrix(df)
m=X.mean(axis=0)
xnew=X-m

X.shape

covariance=(np.cov(xnew.T))
covariance.shape

w, v = LA.eig(covariance)
index=np.argsort(w)[::-1]
eigenv=v[:,index]
eigenv=np.real(eigenv)
value=w[index]
eigenv.shape

pca_5=eigenv[:,0:5]
P = pca_5.T.dot(xnew.T)
X_decomposed1 = np.dot(pca_5.transpose(),xnew.transpose()).transpose()
X_decomposed1.shape

pca_10=eigenv[:,0:10]
P = pca_10.T.dot(xnew.T)
X_decomposed2= np.dot(pca_10.T,xnew.T).transpose()

pca_50=eigenv[:,0:50]
P = pca_50.T.dot(xnew.T)
X_decomposed3= np.dot(pca_50.T,xnew.T).transpose()

pca_100=eigenv[:,0:100]
P = pca_100.T.dot(xnew.T)
X_decomposed4= np.dot(pca_100.T,xnew.T).transpose()

pca_300=eigenv[:,0:300]
P = pca_300.T.dot(xnew.T)
X_decomposed5= np.dot(pca_300.T,xnew.T).transpose()

pca_700=eigenv[:,0:700]
P = pca_700.T.dot(xnew.T)
X_decomposed6= np.dot(pca_700.T,xnew.T).transpose()

"""For sample image 1"""

###PCA components=700

plt.imshow(X[10].reshape(28,28))###Original data visualized
l=[]
plt.show()
mean=X.mean(axis=0)
recovered_data = np.dot(X_decomposed6, eigenv[:,0:700].T) + mean
plt.imshow(recovered_data[10].reshape(28,28))###recovered data from pca visualizedw
plt.show()
plt.imshow((X[10]-recovered_data[10]).reshape(28,28))  ####Subtracted data visualization 
plt.show()
###PCA components=700
diff=X[10]-recovered_data[10]
e1=np.sqrt(np.mean(np.square(diff)))
e1

###PCA components=300

plt.imshow(X[10].reshape(28,28))###Original data visualized
l=[]
plt.show()
mean=X.mean(axis=0)
recovered_data = np.dot(X_decomposed5, eigenv[:,0:300].T) + mean
plt.imshow(recovered_data[10].reshape(28,28))###recovered data from pca visualizedw
plt.show()
plt.imshow((X[10]-recovered_data[10]).reshape(28,28))  ####Subtracted data visualization 
plt.show()
###PCA components=700
diff=X[10]-recovered_data[10]
e2=np.sqrt(np.mean(np.square(diff)))
e2

###PCA components=100

plt.imshow(X[10].reshape(28,28))###Original data visualized
l=[]
plt.show()
mean=X.mean(axis=0)
recovered_data = np.dot(X_decomposed4, eigenv[:,0:100].T) + mean
plt.imshow(recovered_data[10].reshape(28,28))###recovered data from pca visualizedw
plt.show()
plt.imshow((X[10]-recovered_data[10]).reshape(28,28))  ####Subtracted data visualization 
plt.show()
###PCA components=700
diff=X[10]-recovered_data[10]
e3=np.sqrt(np.mean(np.square(diff)))

###PCA components=50

plt.imshow(X[10].reshape(28,28))###Original data visualized
l=[]
plt.show()
mean=X.mean(axis=0)
recovered_data = np.dot(X_decomposed3, eigenv[:,0:50].T) + mean
plt.imshow(recovered_data[10].reshape(28,28))###recovered data from pca visualizedw
plt.show()
plt.imshow((X[10]-recovered_data[10]).reshape(28,28))  ####Subtracted data visualization 
plt.show()
###PCA components=50
diff=X[10]-recovered_data[10]
e4=np.sqrt(np.mean(np.square(diff)))

###PCA components=10

plt.imshow(X[10].reshape(28,28))###Original data visualized
l=[]
plt.show()
mean=X.mean(axis=0)
recovered_data = np.dot(X_decomposed2, eigenv[:,0:10].T) + mean
plt.imshow(recovered_data[10].reshape(28,28))###recovered data from pca visualizedw
plt.show()
plt.imshow((X[10]-recovered_data[10]).reshape(28,28))  ####Subtracted data visualization 
plt.show()
###PCA components=10
diff=X[10]-recovered_data[10]
e5=np.sqrt(np.mean(np.square(diff)))

###PCA components=5

plt.imshow(X[10].reshape(28,28))###Original data visualized
l=[]
plt.show()
mean=X.mean(axis=0)
recovered_data = np.dot(X_decomposed1, eigenv[:,0:5].T) + mean
plt.imshow(recovered_data[10].reshape(28,28))###recovered data from pca visualizedw
plt.show()
plt.imshow((X[10]-recovered_data[10]).reshape(28,28))  ####Subtracted data visualization 
plt.show()
###PCA components=50
diff=X[10]-recovered_data[10]
e6=np.sqrt(np.mean(np.square(diff)))

x_axis=[700,300,100,50,10,5]
error=[e1,e2,e3,e4,e5,e6]
plt.plot(x_axis,error)
error

"""For sample image 2"""

###Pca components=700
error_2=[]
x_axis2=[700,300,100,50,10,5]
plt.imshow(X[15].reshape(28,28))
plt.show()
mean=X.mean(axis=0)
recovered_data = np.dot(X_decomposed6, eigenv[:,0:700].T) + mean
plt.imshow(recovered_data[15].reshape(28,28))
plt.show()
plt.imshow((X[15]-recovered_data[15]).reshape(28,28))
diff=X[15]-recovered_data[15]
err6=np.sqrt(np.mean(np.square(diff)))
error_2.append(err6)

###Pca components=300
plt.imshow(X[15].reshape(28,28))
plt.show()
mean=X.mean(axis=0)
recovered_data = np.dot(X_decomposed5, eigenv[:,0:300].T) + mean
plt.imshow(recovered_data[15].reshape(28,28))
plt.show()
plt.imshow((X[15]-recovered_data[15]).reshape(28,28))
diff=X[15]-recovered_data[15]
err5=np.sqrt(np.mean(np.square(diff)))
error_2.append(err5)

##Pca components=100
plt.imshow(X[15].reshape(28,28))
plt.show()
mean=X.mean(axis=0)
recovered_data = np.dot(X_decomposed4, eigenv[:,0:100].T) + mean
plt.imshow(recovered_data[15].reshape(28,28))
plt.show()
plt.imshow((X[15]-recovered_data[15]).reshape(28,28))
diff=X[15]-recovered_data[15]
err4=np.sqrt(np.mean(np.square(diff)))
error_2.append(err4)

##Pca components=100
plt.imshow(X[15].reshape(28,28))
plt.show()
mean=X.mean(axis=0)
recovered_data = np.dot(X_decomposed3, eigenv[:,0:50].T) + mean
plt.imshow(recovered_data[15].reshape(28,28))
plt.show()
plt.imshow((X[15]-recovered_data[15]).reshape(28,28))
diff=X[15]-recovered_data[15]
err3=np.sqrt(np.mean(np.square(diff)))
error_2.append(err3)

#Pca components=10
plt.imshow(X[15].reshape(28,28))
plt.show()
mean=X.mean(axis=0)
recovered_data = np.dot(X_decomposed2, eigenv[:,0:10].T) + mean
plt.imshow(recovered_data[15].reshape(28,28))
plt.show()
plt.imshow((X[15]-recovered_data[15]).reshape(28,28))
diff=X[15]-recovered_data[15]
err2=np.sqrt(np.mean(np.square(diff)))
error_2.append(err2)

#Pca components=5
plt.imshow(X[15].reshape(28,28))
plt.show()
mean=X.mean(axis=0)
recovered_data = np.dot(X_decomposed1, eigenv[:,0:5].T) + mean
plt.imshow(recovered_data[15].reshape(28,28))
plt.show()
plt.imshow((X[15]-recovered_data[15]).reshape(28,28))
diff=X[15]-recovered_data[15]
err1=np.sqrt(np.mean(np.square(diff)))
error_2.append(err1)

plt.plot(x_axis2,error_2)