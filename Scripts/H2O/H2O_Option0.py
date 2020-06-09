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


# In[ ]:


import sys
print(sys.argv[0]) # prints python_script.py
print(sys.argv[1]) # prints var1


# In[3]:


h2o.init(max_mem_size='32G', log_level='ERRR', jvm_custom_args=['-Xmx32g'])


# In[ ]:


# Load datasets and set parameters


# In[ ]:


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


# In[6]:


#Divide into equal sets of data ~1000 samples
B = np.array_split(df,n)


# In[7]:


#Initialization

#drift_detector = ADWIN()
drift_detector = EDDM()


cum_list = []
for k in range(0,3):   
    cum_list.append(B[k])
train = pd.concat(cum_list)
d = h2o.H2OFrame(train.values)

#Turn into categorical
for col in cat_vars_index:
    d[col] = d[col].asfactor() 


#Train base model RF for stacked ensemble on first 3 batches (required for consistency in stacked ensemble)
start = time.time()

my_rf = H2ORandomForestEstimator(nfolds=10,
                                 fold_assignment='Random',
                                 keep_cross_validation_predictions=True,
				 ntrees=10,
                                 seed=1)
my_rf.train(y=-1, training_frame=d)

#Tune hp of base model GBM for stacked ensemble


#Turn into categorical
for col in cat_vars_index:
    d[col] = d[col].asfactor() 

# GBM hyperparameters search space as in AutoML
gbm_params = {'learn_rate': [0.1],
                #'sample_rate': [i * 0.1 for i in range(5, 11)],
                'col_sample_rate': [0.4, 0.7, 1.0],
                'col_sample_rate_per_tree': [0.4, 0.7, 1.0],
                 #'min_rows': [1, 5, 10, 15, 30, 100],
                'min_split_improvement': [1e-4, 1e-5],
                'ntrees': [10],
		'max_depth':[2]}

# Search criteria - random search and max time for search
search_criteria = {'strategy': 'RandomDiscrete', 'max_runtime_secs': 3600}

# Train and validate a random grid of GBMs (base)
gbm_grid = H2OGridSearch(model=H2OGradientBoostingEstimator(fold_assignment='Random',                                                         
                                                            nfolds=10, 
                                                            keep_cross_validation_predictions=True,
                                                            seed=1),
                          grid_id='gbm_grid',
                          hyper_params=gbm_params,
                          search_criteria=search_criteria)
gbm_grid.train(y=-1, training_frame=d)

my_gbm = []
my_gbm = [gbm_grid.models[k] for k in range(0,len(gbm_grid.models))]

# Train a stacked ensemble using the GBM and GLM above
ensemble1 = H2OStackedEnsembleEstimator(base_models=[my_rf]+[gbm_grid.models[i] for i in range(0,len(gbm_grid.models))])

ensemble1.train(y=-1, training_frame=d)

anytime_model = ensemble1



############################################

#Prequential evaluation

for i in range(3,n):
    
    #Test on next batch for accuracy
    
    test = B[i]

    d = h2o.H2OFrame(test.values)
    
    for col in cat_vars_index:   #Turn into categorical
        d[col] = d[col].asfactor() 
    
    y_hat = anytime_model.predict(d)            #Predict
    predict_array = y_hat['predict'].as_data_frame()
    
    accuracy = sklearn.metrics.accuracy_score(test.iloc[:,-1], predict_array)   
    
    end = time.time()
    print("Test batch %d - Test score %f - Duration %f\n" % (i, accuracy, end-start))
    start = time.time()
    
    #Check drift
    
    drift_in_batch = 0
    for j in range(0,len(B[i])):
        drift_detector.add_element(test.iloc[j,-1] - predict_array.iloc[j][0])
        if drift_detector.detected_change():
            print('Drift in performance detected in batch {} at sample {}'.format(i, j))
            drift_in_batch = 1
    
    if drift_in_batch:
        
        #Train on batches i-3 to i
    
        cum_list.clear()
        for k in range(i-2,i+1):   
            cum_list.append(B[k])
        train = pd.concat(cum_list)
        
        d = h2o.H2OFrame(train.values)
        
        #Turn into categorical
        for col in cat_vars_index:
            d[col] = d[col].asfactor() 
            
        #Train base models

        for k in range(0,len(gbm_grid.models)):
            #training time check
            if (time.time()-start) > 3600:
                continue
            my_gbm[k] = H2OGradientBoostingEstimator(nfolds=10, 
                                                  fold_assignment="Random",
                                                  checkpoint=my_gbm[k], 
                                                  keep_cross_validation_predictions=True,
                                                  seed = i,
                                                  ntrees=10+i,
						  max_depth=2)
            my_gbm[k].train(y=-1, training_frame=d)
        

            
        my_rf = H2ORandomForestEstimator(nfolds=10, 
                                          fold_assignment="Random",
                                          checkpoint=my_rf, 
                                          keep_cross_validation_predictions=True,
                                          ntrees = 10+i,
                                          seed = i)

        my_rf.train(y=-1, training_frame=d)
        
        #training time check
        if (time.time()-start) > 3600:
            continue
        
        #Fit the ensemble
        ensemble2 = H2OStackedEnsembleEstimator(base_models=[my_rf]+my_gbm)
        ensemble2.train(y=-1, training_frame=d)
        anytime_model = ensemble2





