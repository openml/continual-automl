#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h2o 
from h2o.automl import H2OAutoML


# In[2]:


import arff
import numpy as np
import pandas as pd
import time
import sklearn.metrics
import scipy.sparse as sp


# In[3]:


h2o.init()


# In[4]:


alldata = arff.load(open('/home/bcelik/DATA/Forestcover.arff', 'r'),encode_nominal=True)
categorical_columns =[54]


# In[67]:


#alldata = arff.load(open('/home/bilgecelik/Dropbox/PhD/ECML2019/Experiments/DATA/electricity-normalized.arff', 'r'),encode_nominal=True)
#categorical_columns =[1,-1]


# In[72]:


#alldata = arff.load(open('/home/bilgecelik/Dropbox/PhD/ECML2019/Experiments/DATA/AGRAWAL_1M.arff', 'r'),encode_nominal=True)
#categorical_columns =[3,4,5,-1]


# In[73]:


df = pd.DataFrame(alldata["data"])

if pd.isnull(df.iloc[:,:]).any().any():
    print("Data X contains NaN values")
    
df.dropna()


# In[74]:


n = 50
B = np.array_split(df,n)


# In[75]:


#Initialization

aml = H2OAutoML(max_models=20, max_runtime_secs=3600, seed=1, stopping_rounds=2, stopping_tolerance=1e-5)

cum_Xy_list =[]
Xy = B[0]
cum_Xy_list.append(B[0])


start = time.time()
spm = sp.csr_matrix(Xy.values)
d = h2o.H2OFrame(spm)
#Turn into categorical
for col in categorical_columns:
    d[col] = d[col].asfactor() 


aml.train(y=-1, training_frame=d)


#Prequential evaluation

for i in range(1,n):
    
    #Test on next batch for accuracy
    Xy = B[i]
    spm = sp.csr_matrix(Xy.values)
    d = h2o.H2OFrame(spm)
    #Turn into categorical
    for col in categorical_columns:
        d[col] = d[col].asfactor() 
    
    y_hat = aml.predict(d)
    predict_array = y_hat['predict'].as_data_frame().as_matrix()
    
    accuracy = sklearn.metrics.accuracy_score(Xy.iloc[:,-1], predict_array)
    
    end = time.time()
    
    print("Test batch %d - Test score %f - Duration %f\n" % (i, accuracy, end-start))
    
    start = time. time()

    #Train on batches 0 to i
    cum_Xy_list.append(Xy)
    Xy = pd.concat(cum_Xy_list)
    spm = sp.csr_matrix(Xy.values)
    d = h2o.H2OFrame(spm)
    #Turn into categorical
    for col in categorical_columns:
        d[col] = d[col].asfactor() 
    
    aml.train(y=-1, training_frame=d)
    





# In[ ]:




