#!/usr/bin/env python
# coding: utf-8

# # H2O - Option 2 (Warm start / keep training)

# In[1]:


import h2o 
from h2o.automl import H2OAutoML
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.random_forest import H2ORandomForestEstimator


# In[2]:


import arff
import numpy as np
import pandas as pd
import sklearn
import time
import sklearn.metrics as skm
import scipy.sparse as sp

from skmultiflow.data import FileStream
from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection import EDDM


# In[3]:


h2o.init(min_mem_size='10G', max_mem_size='100G')


# In[4]:


# Load datasets and set parameters


# In[5]:


#Airlines
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
df


# In[37]:


#Divide into equal sets of data ~1000 samples
B = np.array_split(df,n)


# In[38]:


#Initialization

#drift_detector = ADWIN()
drift_detector = EDDM()


cum_list = []
for k in range(0,n):
    cum_list.append(B[k])
        
train = pd.concat(cum_list)
spm = sp.csr_matrix(train.values)
d = h2o.H2OFrame(spm)
#Turn into categorical
for col in cat_vars_index:
    d[col] = d[col].asfactor() 
    
start = time.time()

#Train base models for stacked ensemble

my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",
                                      nfolds = 10,
                                      ntrees = 5,
                                      keep_cross_validation_predictions=True,
                                      seed=1)

my_gbm.train(y=-1, training_frame=d)

my_rf = H2ORandomForestEstimator(nfolds=10,
                                 ntrees=5,
                                 keep_cross_validation_predictions=True,
                                 seed=1)
my_rf.train(y=-1, training_frame=d)


# Train a stacked ensemble using the GBM and GLM above
ensemble = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf])

ensemble.train(y=-1, training_frame=d)

anytime_model = ensemble

#Prequential evaluation

for i in range(3,10):
    
    #Test on next batch for accuracy
    
    test = B[i]

    spm = sp.csr_matrix(test.values)
    d = h2o.H2OFrame(spm)
    
    for col in cat_vars_index:   #Turn into categorical
        d[col] = d[col].asfactor() 
    
    y_hat = ensemble.predict(d)            #Predict
    predict_array = y_hat['predict'].as_data_frame().as_matrix()
    
    accuracy = sklearn.metrics.accuracy_score(test.iloc[:,-1], predict_array)   
    
    end = time.time()
    
    print("Test batch %d - Test score %f - Duration %f\n" % (i, accuracy, end-start))
    
    #Check drift
    
    drift_in_batch = 0
    for j in range(0,len(B[i])):
        drift_detector.add_element(test.iloc[j,-1] - predict_array[j][0])
        if drift_detector.detected_change():
            print('Drift in performance detected in batch {} at sample {}'.format(i, j))
            drift_in_batch = 1
    
    if drift_in_batch:
        #Train on batches i-2 to i (last 3 batches)
        cum_list = []
        for k in range(i-2,i+1):
            cum_list.append(B[k])
        
        train = pd.concat(cum_list)
        spm = sp.csr_matrix(train.values)
        d = h2o.H2OFrame(spm)
        #Turn into categorical
        for col in cat_vars_index:
            d[col] = d[col].asfactor() 
    
        my_gbm = H2OGradientBoostingEstimator(distribution='bernoulli', 
                                              nfolds=10, 
                                              ntrees=(5+i), 
                                              checkpoint=my_gbm, 
                                              keep_cross_validation_predictions=True,
                                              seed = i)
        my_gbm.train(y=-1, training_frame=d)
            
        my_rf = H2ORandomForestEstimator(nfolds=10, 
                                         ntrees=(5+i), 
                                         checkpoint=my_rf, 
                                         keep_cross_validation_predictions=True,
                                         seed = i)
        my_rf.train(y=-1, training_frame=d)
        
        ensemble = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf])
        ensemble.train(y=-1, training_frame=d)


# In[ ]:




