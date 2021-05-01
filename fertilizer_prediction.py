#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


dataset=pd.read_csv(r"C:\Users\sai kiran\Downloads\internship files\prjoect dataset\fertilizer4.csv")
dataset.head(2)


# In[4]:


dataset.isnull().any()


# In[5]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset['Soil Type'] =le.fit_transform(dataset['Soil Type'])
dataset['Crop Type'] =le.fit_transform(dataset['Crop Type'])
dataset['Fertilizer Name'] =le.fit_transform(dataset['Fertilizer Name'])


# In[6]:


dataset.head()


# In[7]:


x=dataset.iloc[:,:8].values
x.shape


# In[8]:


y=dataset.iloc[:,-1].values
y


# In[9]:


x[:,3].max()


# In[10]:


from sklearn.preprocessing import OneHotEncoder
one1 =OneHotEncoder(categorical_features= [3])
x=one1.fit_transform(x).toarray()


# In[11]:


x=x[:,1:]


# In[12]:


x[:,7].max()


# In[13]:


from sklearn.preprocessing import OneHotEncoder
one2 =OneHotEncoder(categorical_features= [7])
x=one2.fit_transform(x).toarray()


# In[14]:


x=x[:,1:]


# In[15]:


y[:].max()


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[17]:


x_train.shape


# In[18]:


x_test.shape


# In[19]:


y_train.shape


# In[20]:


y_test.shape


# In[21]:


# Fitting Naive Bayes to the Training set
'''from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)'''


# In[22]:


# Fitting KNN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) #metric is accuracy n p is 2cm
classifier.fit(x_train, y_train)


# In[23]:


# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred


# In[24]:


y_test


# In[25]:


#from sklearn.metrics import r2_score
#r2_score(y_test,y_pred)
#print(y_test,y_pred)


# In[26]:


ypred1=classifier.predict([[0,0,1,0,0,0,1,0,0,0,0,0,0,1,26,65,38,37,14,0]])
ypred1


# In[27]:


if (ypred1==0):
    print("10-26-26")
elif (ypred1==1):
    print("14-35-14")
elif (ypred1==2):
    print("17-17-17")
elif (ypred1==3):
    print("20-20")
elif (ypred1==4):
    print("28-28")
elif (ypred1==5):
    print("DAP")
elif (ypred1==6):
    print("Urea")
    
  


# In[28]:


from sklearn.metrics import accuracy_score


# In[30]:


pred=accuracy_score(y_test,y_pred)
pred


# In[ ]:




