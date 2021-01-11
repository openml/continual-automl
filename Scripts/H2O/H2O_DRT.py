#!/usr/bin/env python
# coding: utf-8

# # H2O - Option 1 (Best model adaptation after drift)

# In[1]:


import h2o 
from h2o.automl import H2OAutoML
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator


# In[2]:


import arff
import numpy as np
import pandas as pd
import time
import sklearn
import sklearn.metrics
import scipy.sparse as sp

from skmultiflow.data import FileStream
from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection import EDDM


# In[3]:


h2o.init()


# In[4]:


#Airlines
alldata = arff.load(open('/DATA/Real/new_airlines.arff', 'r'),encode_nominal=True)
categorical_columns =[0, 2, 3, 4, -1]
target_moved = 0
n=540
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


# In[6]:


#Divide into equal sets of data ~1000 samples
B = np.array_split(df,n)


# In[7]:


#Initialization

aml = H2OAutoML(max_runtime_secs=60, keep_cross_validation_predictions = True, exclude_algos = ["GLM"])
#drift_detector = ADWIN()
drift_detector = EDDM()

cum_list = []
for k in range(0,3):    
    cum_list.append(B[k])

train = pd.concat(cum_list)
spm = sp.csr_matrix(train.values)
d = h2o.H2OFrame(spm)
#Turn into categorical
for col in cat_vars_index:
    d[col] = d[col].asfactor() 

start = time.time()

aml.train(y=-1, training_frame=d)
anytime_model = aml


#Prequential evaluation

for i in range(3, 4):
    
    #Test on next batch for accuracy
    
    test = B[i]

    spm = sp.csr_matrix(test.values)
    d = h2o.H2OFrame(spm)
    
    #Turn into categorical

    for col in cat_vars_index:
        d[col] = d[col].asfactor() 
    
    y_hat = anytime_model.predict(d)
    predict_array = y_hat['predict'].as_data_frame()
    
    accuracy = sklearn.metrics.accuracy_score(test.iloc[:,-1], predict_array.iloc[:,0])
    
    end = time.time()
    
    print("Test batch %d - Test score %f - Duration %f\n" % (i, accuracy, end-start))
    
    
    drift_in_batch = 0
    for j in range(0,len(B[i])):
        drift_detector.add_element(test.iloc[j,-1] - predict_array.iloc[j,0])
        if drift_detector.detected_change():
            print('Drift in performance detected at batch {}, sample {}'.format(i, j))
            drift_in_batch = 1
    
    if drift_in_batch:
        #Train on last three batches
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
            
        #Retrain the previous leader model with new data
        
        if aml.leader.algo == 'stackedensemble':     #If it is stacked ensemble, fix base models and fix the batch siz
            b_models = aml.leader.full_parameters['base_models']
            base = [b_models['actual_value'][i]['name'] for i in range(0,len(b_models['actual_value']))]
            
            #Here we retrain base models before calling stack ensemble
            print("stacked")          
            for b_model in base:
                if 'GLM' in b_model:       #GLM is giving error with re-training
                    base.remove(b_model)
                else:
                    m = h2o.get_model(b_model)
                    m.train(y=-1, training_frame=d)
            
            ensemble = H2OStackedEnsembleEstimator(base_models=base)
            ensemble.train(y=-1, training_frame=d)
            anytime_model = ensemble
            
        else:
            aml.leader.train(y=-1, training_frame=d)
            anytime_model = aml.leader
        
       

# In[27]:


from h2o.estimators.gbm import H2OGradientBoostingEstimator

m = h2o.get_model('GBM_grid__1_AutoML_20200518_140119_model_4')
print(m)


# In[28]:


m_new = H2OGradientBoostingEstimator(checkpoint = m_new)
m_new.train(y=-1, training_frame=d)
print(m_new)

