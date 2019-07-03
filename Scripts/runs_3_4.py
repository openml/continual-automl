#!/usr/bin/env python
# coding: utf-8

# In[14]:


import h2o 
from h2o.automl import H2OAutoML


# In[24]:


import arff
import numpy as np
import pandas as pd
import time
import sklearn.metrics
import scipy.sparse as sp


# In[16]:


h2o.init()


# In[17]:


alldata = arff.load(open('/home/bcelik/DATA/HYPERPLANE_500k.arff', 'r'),encode_nominal=True)


# In[18]:


df = pd.DataFrame(alldata["data"])

if pd.isnull(df.iloc[:,:]).any().any():
    print("Data X contains NaN values")
    
df.dropna()


# In[51]:


n = 50
B = np.array_split(df,n)

binary=0
if max(B[0].iloc[:,-1])==1:
    binary=1


# In[74]:


#Initialization

aml = H2OAutoML(max_models=20, max_runtime_secs=3600, seed=1, stopping_rounds=2, stopping_tolerance=1e-5)

cum_Xy_list =[]

for i in range(0,31):
	cum_Xy_list.append(B[i])
Xy = pd.concat(cum_Xy_list)

start = time.time()
spm = sp.csr_matrix(Xy.values)
d = h2o.H2OFrame(spm)
if binary==1:
    d[-1] = d[-1].asfactor()


aml.train(y=-1, training_frame=d)


#Prequential evaluation

for i in range(31,n):
    
    #Test on next batch for accuracy
    Xy = B[i]
    spm = sp.csr_matrix(Xy.values)
    d = h2o.H2OFrame(spm)
    if binary==1:
        d[-1] = d[-1].asfactor()
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
    if binary==1:
        d[-1] = d[-1].asfactor()
    
    aml.train(y=-1, training_frame=d)
    





# In[ ]:




