#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv(r"C:\Users\sai kiran\Desktop\fertilizer4.csv")
dataset.head()

dataset.isnull().any()


# In[3]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset['Soil Type'] =le.fit_transform(dataset['Soil Type'])
dataset['Crop Type'] =le.fit_transform(dataset['Crop Type'])
dataset['Fertilizer Name'] =le.fit_transform(dataset['Fertilizer Name'])


# In[4]:


dataset.head()
x=dataset.iloc[:,:8].values
x
y=dataset.iloc[:,8:9].values
y

x[:,3].max()

from sklearn.preprocessing import OneHotEncoder
one1 =OneHotEncoder(categorical_features= [3])
x=one1.fit_transform(x).toarray()
x=x[:,1:]


# In[5]:


x[:,7].max()
from sklearn.preprocessing import OneHotEncoder
one2 =OneHotEncoder(categorical_features= [7])
x=one2.fit_transform(x).toarray()
x=x[:,1:]


# In[6]:


from sklearn.preprocessing import OneHotEncoder
one5 =OneHotEncoder(categorical_features= [0])
y=one5.fit_transform(y).toarray()



# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[8]:


# Fitting Naive Bayes to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) #metric is accuracy n p is 2cm
classifier.fit(x_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(x_test)

    
  



# In[9]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test[:,0], y_pred[:,0])
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test[:,1], y_pred[:,1])
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test[:,2], y_pred[:,2])
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test[:,3], y_pred[:,3])
from sklearn.metrics import confusion_matrix
cm5 = confusion_matrix(y_test[:,4], y_pred[:,4])
from sklearn.metrics import confusion_matrix
cm7 = confusion_matrix(y_test[:,6], y_pred[:,6])
from sklearn.metrics import confusion_matrix
cm6 = confusion_matrix(y_test[:,5], y_pred[:,5])


# In[10]:


# Visualising the Training set results
import sklearn.metrics as metrics
fpr1, tpr1, threshold = metrics.roc_curve(y_test[:,0],y_pred[:,0])
roc_auc1 = metrics.auc(fpr1,tpr1)
import sklearn.metrics as metrics
fpr2, tpr2, threshold = metrics.roc_curve(y_test[:,1],y_pred[:,1])
roc_auc2 = metrics.auc(fpr2,tpr2)
import sklearn.metrics as metrics
fpr3, tpr3, threshold = metrics.roc_curve(y_test[:,2],y_pred[:,2])
roc_auc3 = metrics.auc(fpr3,tpr3)
import sklearn.metrics as metrics
fpr4, tpr4, threshold = metrics.roc_curve(y_test[:,3],y_pred[:,3])
roc_auc4 = metrics.auc(fpr4,tpr4)
import sklearn.metrics as metrics
fpr5, tpr5, threshold = metrics.roc_curve(y_test[:,4],y_pred[:,4])
roc_auc5 = metrics.auc(fpr5,tpr5)
import sklearn.metrics as metrics
fpr6, tpr6, threshold = metrics.roc_curve(y_test[:,5],y_pred[:,5])
roc_auc6 = metrics.auc(fpr6,tpr6)
import sklearn.metrics as metrics
fpr7, tpr7, threshold = metrics.roc_curve(y_test[:,6],y_pred[:,6])
roc_auc7 = metrics.auc(fpr7,tpr7)


# In[11]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.2f' % roc_auc1)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[12]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr2, tpr2, 'b', label = 'AUC = %0.2f' % roc_auc2)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[13]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr3, tpr3, 'b', label = 'AUC = %0.2f' % roc_auc3)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[14]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr4, tpr4, 'b', label = 'AUC = %0.2f' % roc_auc4)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[15]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr5, tpr5, 'b', label = 'AUC = %0.2f' % roc_auc5)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[16]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr6, tpr6, 'b', label = 'AUC = %0.2f' % roc_auc6)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[17]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr7, tpr7, 'b', label = 'AUC = %0.2f' % roc_auc7)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:





# In[ ]:




