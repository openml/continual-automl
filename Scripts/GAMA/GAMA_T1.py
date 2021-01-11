#!/usr/bin/env python
# coding: utf-8

# # GAMA Option-5  (Do nothing - train once)

# In[1]:


import arff
import numpy as np
import pandas as pd
import time

import sklearn.metrics
from gama import GamaClassifier
from gama.postprocessing import EnsemblePostProcessing


# In[2]:


import sys
print(sys.argv[0]) # prints python_script.py
print(sys.argv[1]) # prints var1


# In[3]:


sys.argv[1]=1


# In[4]:


if int(sys.argv[1]) == 1:
    #airlines
    alldata = arff.load(open('/DATA/Real/new_airlines.arff', 'r'),encode_nominal=True)
    categorical_columns =[0, 2, 3, 4, -1]
    target_moved = 0
    n=540
    cat_limit = 15


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

#Divide into equal sets of data ~20,000 samples
B = np.array_split(df,n)

B[0]


# In[6]:


#Initialization

cls = GamaClassifier(max_total_time=3600, keep_analysis_log=None, n_jobs=1, scoring='accuracy', post_processing_method=EnsemblePostProcessing())

X = B[0].iloc[:,0:-1]
y = B[0].iloc[:,-1]



print("Starting `fit`")
cls.fit(X, y)

anytime_model = cls

#Prequential evaluation

for i in range(1,n):
    
    #Test on next batch for accuracy
    X = B[i].iloc[:,0:-1]
    y = B[i].iloc[:,-1]
    y_hat = cls.predict(X)
    accuracy = sklearn.metrics.accuracy_score(y, y_hat)
    print("Test batch %d - Test score %f\n" % (i, accuracy))
    


# In[ ]:




