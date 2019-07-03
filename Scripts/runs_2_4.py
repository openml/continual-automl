#!/usr/bin/env python
# coding: utf-8

# In[1]:


import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import arff
import numpy as np
import pandas as pd
import time


# In[2]:


alldata = arff.load(open('/home/bcelik/DATA/HYPERPLANE_500k.arff', 'r'),encode_nominal=True)


# In[3]:


df = pd.DataFrame(alldata["data"])

if pd.isnull(df.iloc[:,:]).any().any():
    print("Data X contains NaN values")
    
df.dropna()


# In[63]:


n = 50
B = np.array_split(df,n)


# In[ ]:


#Initialization

cum_X_list =[]
cum_y_list =[]

cls = autosklearn.classification.AutoSklearnClassifier(ml_memory_limit=6144, time_left_for_this_task=3600)

X = B[0].iloc[:,0:-1]
y = B[0].iloc[:,-1]
cum_X_list.append(B[0].iloc[:,0:-1])
cum_y_list.append(B[0].iloc[:,-1])

start = time.time()
cls.fit(X, y)

#Prequential evaluation

for i in range(1,n):
    
    #Test on next batch for accuracy
    X = B[i].iloc[:,0:-1]
    y = B[i].iloc[:,-1]
    y_hat = cls.predict(X)
    accuracy = sklearn.metrics.accuracy_score(y, y_hat)
    end = time.time()
    print("Test batch %d - Test score %f - Duration %f\n" % (i, accuracy, end-start))
    start = time. time()
    
    #Train on batches 0 to i
    cum_X_list.append(X)
    cum_y_list.append(y)
    X = pd.concat(cum_X_list)
    y = pd.concat(cum_y_list)
    
    cls.fit(X, y)
    
    


# In[ ]:




