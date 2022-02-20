#!/usr/bin/env python
# coding: utf-8

# **Question1(i)**

# Model :Nominal,
# 
# Type :Ordinal
# 
# Max.Price:Ratio
# 
# Airbag:Ordinal

# **Question1(ii)**

# In[62]:


import pandas as pd ##Importing libraries
import numpy as np


# In[63]:


df=pd.read_csv("Cars93.csv")  ##reading csv file
df.head()


# In[64]:


df.loc[:, df.isnull().any()].columns ###Columns with NA values


# In[65]:


df['Rear.seat.room'].isna().sum()  ##No. of null values in the coloumn


# In[66]:


df['Luggage.room'].isna().sum()    ####No. of null values in the coloumn


# In[67]:


from sklearn.impute import SimpleImputer  ###Library to import Impute the missing values


# In[68]:


impute=SimpleImputer(strategy='mean')  ##calling the imputer


# In[69]:


df[['Rear.seat.room','Luggage.room']]


# In[70]:


imputed_df=impute.fit_transform(df[['Rear.seat.room','Luggage.room']]) ###Replacing the value missing values with the mean 


# In[71]:


newdf=pd.DataFrame(imputed_df,columns=['Rear.seat.room','Luggage.room']) ####new data frame with no NA values


# In[72]:


newdf.head()


# In[73]:


df['Rear.seat.room']=newdf['Rear.seat.room']   ##replacing the the empty columns in original data frame with non empty columns


# In[74]:


df['Luggage.room']=newdf['Luggage.room']        ##replacing the the empty columns in original data frame with non empty columns


# ***Question1(iii)***

# In[75]:



noisy_data=df.drop(['Manufacturer','Model','Type','AirBags','DriveTrain','Man.trans.avail','Origin'],axis=1) ###creating a data frae with no strings
a=noisy_data.drop(56)  ###dropping the 56th row for now because of a string rotary present 
a['Cylinders']=a.Cylinders.astype(float)  ## changing the dtype of 'Cylinders' from object to float 
b=np.median(a['Cylinders']) ###finding the median of 'Cylinders' columns
c=str(b) ###Since dtype 'Cylinders' in noisy_data is object hence converting b to string dtype
data=noisy_data.replace({'Cylinders':'rotary'},c) ###replacing the word rotary from the 'Cylinders' columns with the median of the columns
data['Cylinders']=data.Cylinders.astype(float) ###Changing the dtype of 'Cylinders' from object to float
data['Cylinders']=data.Cylinders.astype(int) #### changing the dtype of 'Cylinders' from float to int


# In[76]:


for column in data.columns:   ###for loop to read the columns
    a1=np.mean(data[column])  ### to calculate mean of every column in the loop
    a2=np.std(data[column])   ### to calculate std of every column in the loop
    for i in data[column]:    ###for loop for some ith value of s particular column
       if i>a1+3*a2 or i==a1+3*a2:    ### condition for some ith value being a outlier that is if the ith value of a column is greater than or equal to 3 standard deviations after mean then it is an outlier
         if data[column].dtypes==float:  ###if dtype is float replace the outlier value with mean of the column    
                data.replace({column:i},np.mean(noisy_data[column]))
         else:
                data.replace({column:i},np.median(noisy_data[column]))### if dtype is int replace the outlier value with median of the column
                
         


# In[77]:


####replacing the original columns with denoised column
df['Min.Price']=data['Min.Price']  
df['Price']=data['Price']
df['Max.Price']=data['Max.Price']
df['MPG.city']=data['MPG.city']
df['MPG.highway']=data['MPG.highway']
df['Cylinders']=data['Cylinders']
df['EngineSize']=data['EngineSize']
df['Horsepower']=data['Horsepower']
df['RPM']=data['RPM']
df['Rev.per.mile']=data['Rev.per.mile']
df['Passengers']=data['Passengers']
df['Length']=data['Length']
df['Wheelbase']=data['Wheelbase']
df['Width']=data['Width']
df['Turn.circle']=data['Turn.circle']
df['Rear.seat.room']=data['Rear.seat.room']
df['Luggage.room']=data['Luggage.room']
df['Weight']=data['Weight']



# ***Question1(vi)***
# 

# In[78]:


from sklearn.model_selection import train_test_split ###importing library to train,validate,test


# In[79]:


X=df.drop(['Price'],axis=1)
y=df['Price']


# In[80]:


X_train1,X_test,y_train1,y_test=train_test_split(X,y,train_size=0.9) ### 90% training dataset 10% test dataset


# In[81]:


X_val,X_train,y_val,y_train=train_test_split(X_train1,y_train1,train_size=0.2) ### out of 90% training 20% validation


# **Question1(iv)**

# In[82]:


from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


# In[96]:


ohe=OneHotEncoder(sparse=False)
df["Model"].unique()


# In[97]:


ohe_df=ohe.fit_transform(df[['Manufacturer','Model']])
ohe_df.shape
dataframe=pd.DataFrame(ohe_df,columns=['Acura','Audi','BMW','Buick','Cadillac','Chevrolet','Chrylser','Chrysler','Dodge','Eagle','Ford','Geo','Honda','Hyundai','Infiniti','Lexus','Lincoln','Mazda','Mercedes-Benz','Mercury','Mitsubishi','Nissan','Oldsmobile','Plymouth','Pontiac','Saab','Saturn','Subaru','Suzuki','Toyota','Volkswagen','Volvo','Integra', 'Legend', '90', '100', '535i', 'Century', 'LeSabre','Roadmaster', 'Riviera', 'DeVille', 'Seville', 'Cavalier','Corsica', 'Camaro', 'Lumina', 'Lumina_APV', 'Astro', 'Caprice','Corvette', 'Concorde', 'LeBaron', 'Imperial', 'Colt', 'Shadow','Spirit', 'Caravan', 'Dynasty', 'Stealth', 'Summit', 'Vision','Festiva', 'Escort', 'Tempo', 'Mustang', 'Probe', 'Aerostar','Taurus', 'Crown_Victoria', 'Metro', 'Storm', 'Prelude', 'Civic', 'Accord', 'Excel', 'Elantra', 'Scoupe', 'Sonata', 'Q45', 'ES300','SC300', 'Continental', 'Town_Car', '323', 'Protege', '626', 'MPV','RX-7', '190E', '300E', 'Capri', 'Cougar', 'Mirage', 'Diamante','Sentra', 'Altima', 'Quest', 'Maxima', 'Achieva', 'Cutlass_Ciera','Silhouette', 'Eighty-Eight', 'Laser', 'LeMans', 'Sunbird','Firebird', 'Grand_Prix', 'Bonneville', '900', 'SL', 'Justy','Loyale', 'Legacy', 'Swift', 'Tercel', 'Celica', 'Camry', 'Previa','Fox', 'Eurovan', 'Passat', 'Corrado', '240', '850'])


# In[85]:


oe=OrdinalEncoder()


# In[86]:


encoded_df=oe.fit_transform(df[['Type','DriveTrain','Origin','Man.trans.avail','AirBags']])


# In[87]:


df1=pd.DataFrame(encoded_df,columns=['Type','DriveTrain','Origin','Man.trans.avail','AirBags'])


# In[100]:


final=pd.concat([df,dataframe],axis=1)
final['Type']=df1['Type']
final['DriveTrain']=df1['DriveTrain']
final['Origin']=df1['Origin']
final['Man.trans.avail']=df1['Man.trans.avail']
final['AirBags']=df1['AirBags']

final


# In[99]:





# **Question(v)**

# In[91]:


from sklearn.preprocessing import StandardScaler


# In[92]:


ss=StandardScaler()


# In[93]:


X=df.iloc[:,[13,14]]
normalized_df=ss.fit_transform(X)


# In[94]:


df2=pd.DataFrame(normalized_df,columns=['RPM','Rev.per.mile'])
df['RPM']=df2['RPM']
df['Rev.per.mile']=df2['Rev.per.mile']


# In[95]:


df


# In[ ]:





# In[ ]:




