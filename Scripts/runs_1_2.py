#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tpot
import arff
import numpy as np
import pandas as pd
import time


# In[2]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier


# In[3]:


#alldata = arff.load(open('/home/bcelik/DATA/electricity-normalized.arff', 'r'),encode_nominal=True)

alldata = arff.load(open('/home/bcelik/DATA/Forestcover.arff', 'r'),encode_nominal=True)


# In[17]:


df = pd.DataFrame(alldata["data"])

if pd.isnull(df.iloc[:,:]).any().any():
    print("Data X contains NaN values")
    
df.dropna()


# In[18]:


n = 50
B = np.array_split(df,n)


# In[ ]:


pipeline_optimizer = TPOTClassifier(generations = 100, warm_start = True, verbosity=0, max_time_mins=60, early_stop = 10)
f= open("Test_scores.txt","a+")

#Initialization

X = B[0].iloc[:,0:-1]
y = B[0].iloc[:,-1]

start = time.time()
pipeline_optimizer.fit(X, y)


for i in range(1,n):
    X = B[i].iloc[:,0:-1]
    y = B[i].iloc[:,-1]
    accuracy = pipeline_optimizer.score(X, y)
    end = time.time()
    f.write("Test batch %d - Test score %f - Duration %f\n" % (i, accuracy, end-start))
    print("Test batch %d - Test score %f - Duration %f\n" % (i, accuracy, end-start))
    #file  = 'tpot_exported_pipelinefor' + str(i) + '.py'
    #pipeline_optimizer.export(file)  
    start = time. time()
    pipeline_optimizer.fit(X, y)
    

f.close()
    


# In[ ]:




