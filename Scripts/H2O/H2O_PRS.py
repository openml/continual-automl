#!/usr/bin/env python
# coding: utf-8

# # H2O - Option 4 (Train with every batch)

# In[1]:


import h2o 
from h2o.automl import H2OAutoML


# In[2]:


import arff
import numpy as np
import pandas as pd
import time
import sklearn
import sklearn.metrics
import scipy.sparse as sp


# In[6]:


import sys
print(sys.argv[0]) # prints python_script.py
print(sys.argv[1]) # prints var1


# In[7]:


h2o.init()


# In[47]:


if int(sys.argv[1]) == 1:
    #airlines
    alldata = arff.load(open('/DATA/Real/new_airlines.arff', 'r'),encode_nominal=True)
    categorical_columns =[0, 2, 3, 4, -1]
    target_moved = 0
    n=27
    cat_limit = 5


# Preprocessing of data: Drop NaNs, move target to the end, remove high cardinality categorical variables (h2o)

if target_moved == 1:
    df = pd.DataFrame(alldata["data"])
    columns = df.columns.values.tolist()
    columns.remove(0)
    columns.append(0)
    df = df.reindex(columns, axis=1) 

else:
    df = pd.DataFrame(alldata["data"])

if pd.isnull(df.iloc[:,:]).any().any():
    print("Data X contains NaN values")
    
df.dropna()

#Remove high cardinality categorical variables because of the bug in H2O
to_remove=[]

for i in categorical_columns:
    if df.iloc[:,i].max() > cat_limit:    #Limit for cardinality in categorical variables - 17 gave an error)
        to_remove.append(i)

df.drop(to_remove , axis='columns', inplace=True)

for i in to_remove:
    categorical_columns.remove(i)

cat_vars_index = [df.columns.get_loc(c) for c in categorical_columns if c in df]
if target_moved == 0:
    cat_vars_index.append(-1)

df.iloc[:,-1].replace(0, 2, inplace = True)


# In[49]:


#Divide into equal sets of data ~20,000 samples
B = np.array_split(df,n)


# In[7]:


#Initialization

aml = H2OAutoML(max_runtime_secs=3600)


train = B[0]

start = time.time()

spm = sp.csr_matrix(train.values)
d = h2o.H2OFrame(spm)

#Turn into categorical
for col in cat_vars_index:
    d[col] = d[col].asfactor() 


aml.train(y=-1, training_frame=d)
anytime_model = aml.leader

#Prequential evaluation

for i in range(1,n):
    
    #Test on next batch for accuracy
    
    test = B[i]

    spm = sp.csr_matrix(test.values)
    d = h2o.H2OFrame(spm)
    
    #Turn into categorical
    for col in cat_vars_index:
        d[col] = d[col].asfactor() 
    
    y_hat = anytime_model.predict(d)
    predict_array = y_hat['predict'].as_data_frame().as_matrix()
    
    accuracy = sklearn.metrics.accuracy_score(test.iloc[:,-1], predict_array)
    
    end = time.time()
    
    print("Test batch %d - Test score %f - Duration %f\n" % (i, accuracy, end-start))
    

    #Train on batches i-3 to i
    
    cum_list = []
    for k in range(i-2,i+1):
        
        if k < 0 :
            k = 0
        cum_list.append(B[k])
        
    train = pd.concat(cum_list)
    spm = sp.csr_matrix(train.values)
    d = h2o.H2OFrame(spm)
    #Turn into categorical
    for col in cat_vars_index:
        d[col] = d[col].asfactor() 
    
    aml.train(y=-1, training_frame=d)
    anytime_model = aml.leader




# In[ ]:




