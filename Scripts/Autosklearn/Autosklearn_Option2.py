#!/usr/bin/env python
# coding: utf-8

# # Auto-sklearn Option 2 (Keep training / warm start) 

# In[131]:


import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import multiprocessing
import shutil

from autosklearn.metrics import accuracy
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import MULTICLASS_CLASSIFICATION

tmp_folder = '/tmp/option2_'+ str(sys.argv[1]) + '_tmp'
output_folder = '/tmp/option2_' + str(sys.argv[1]) + '_out'

for dir_ in [tmp_folder, output_folder]:
    try:
        shutil.rmtree(dir_)
    except OSError:
        pass

import arff
import numpy as np
import pandas as pd
import time

from skmultiflow.data import FileStream
from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection import EDDM


# In[132]:


import sys
print(sys.argv[0]) # prints python_script.py
print(sys.argv[1]) # prints var1


# In[134]:


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


# In[136]:


#Divide into equal sets of data ~1,000 samples
B = np.array_split(df,n)


# In[137]:


B[0]


# In[141]:


#Initial set up of (parallel) run classifier   

def get_spawn_classifier(X_train, y_train):
    def spawn_classifier(seed):
        """Spawn a subprocess.

        auto-sklearn does not take care of spawning worker processes. This
        function, which is called several times in the main block is a new
        process which runs one instance of auto-sklearn.
        """

        # Use the initial configurations from meta-learning only first batch.
        if seed == 0:
            initial_configurations_via_metalearning = 25
            smac_scenario_args = {}
        else:
            initial_configurations_via_metalearning = 0 #Setting the initial configurations found by meta-learning to zero makes auto-sklearn use the regular SMAC algorithm for suggesting new hyperparameter configurations.
            smac_scenario_args = {'initial_incumbent': 'RANDOM'} #What does RANDOM do instead of DEFAULT from PCS?

        # Arguments which are different to other runs of auto-sklearn:
        # 1. all classifiers write to the same output directory
        # 2. shared_mode is set to True, this enables sharing of data between
        # models.
        # 3. all instances of the AutoSklearnClassifier must have a different seed!
        automl = AutoSklearnClassifier(
            time_left_for_this_task=2400,    #2/3 of 1 hour spared for optimization run
            ml_memory_limit=6144,
            shared_mode=True,  # tmp folder will be shared between seeds
            tmp_folder=tmp_folder,
            output_folder=output_folder,
            delete_tmp_folder_after_terminate=False,
            delete_output_folder_after_terminate=False,
            ensemble_size=0,    # ensembles will be built after optimization run is finished
            initial_configurations_via_metalearning=(
                initial_configurations_via_metalearning
            ),
            seed=seed,
            smac_scenario_args=smac_scenario_args,
        )
        automl.fit(X_train, y_train)
    return spawn_classifier


# In[154]:


#Run

#Initialization

cum_X_list =[]
cum_y_list =[]

#drift_detector = ADWIN()
drift_detector = EDDM()

#Initialize with first batch
for k in range(0,1):
    cum_X_list.append(B[k].iloc[:,0:-1])
    cum_y_list.append(B[k].iloc[:,-1])
            
X_train = pd.concat(cum_X_list)
y_train = pd.concat(cum_y_list)
        

#Start spawn classifier
processes = []
spawn_cls = get_spawn_classifier(X_train, y_train)

#Join processes
p = multiprocessing.Process(
        target=spawn_cls,
        args=(0,)   #initial seed
        )
p.start()
processes.append(p)
for p in processes:
    p.join()

#Build an ensemble 
cls = AutoSklearnClassifier(
        time_left_for_this_task=1200,    #1/3 of 1 hour spared for fitting ensemble
        ml_memory_limit=6144,
        shared_mode=True,
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False,
        initial_configurations_via_metalearning=0,
        seed=0,
    )

#Fit ensemble, change size and nbest if necessary
cls.fit_ensemble(
        y_train,
        task=MULTICLASS_CLASSIFICATION,
        metric=accuracy,
    )

anytime_model = cls


#Prequential evaluation

for i in range(24,27):
    
    #Test on next batch for accuracy
    X_test = B[i].iloc[:,0:-1]
    y_test = B[i].iloc[:,-1]
    y_test = y_test.to_numpy()
    y_hat = cls.predict(X_test)
    
    print("y_hat")
    print(y_hat)
    print("y_test")
    print(y_test)
    
    accuracy_value = sklearn.metrics.accuracy_score(y_test, y_hat)
    print("Test batch %d - Test score %f\n" % (i, accuracy_value))
    
    #Check for drift
    drift_in_batch = 0
    for j in range(0,len(B[i])):
        drift_detector.add_element(y_test[j] - y_hat[j])
        if drift_detector.detected_change():
            print('Drift in performance detected at sample {}'.format(j))
            drift_in_batch = 1
     
    if drift_in_batch:
        #Train on batches i-3 to i
        cum_X_list = []
        cum_y_list = []
        for k in range(i-2,i+1):
        
            if k < 0 :
                k = 0
            cum_X_list.append(B[k].iloc[:,0:-1])
            cum_y_list.append(B[k].iloc[:,-1])
            
        X_train = pd.concat(cum_X_list)
        y_train = pd.concat(cum_y_list)
        
        #Start spawn classifier
        processes = []
        spawn_cls = get_spawn_classifier(X_train, y_train)

        #Join processes
        p = multiprocessing.Process(
            target=spawn_cls,
            args=(i,)   #initial seed
            )
        p.start()
        processes.append(p)
        for p in processes:
            p.join()

        #Build an ensemble 
        cls = AutoSklearnClassifier(
                time_left_for_this_task=1200,   #1/3 of 1 hour spared for fitting ensemble
                ml_memory_limit=6144,
                shared_mode=True,
                tmp_folder=tmp_folder,
                output_folder=output_folder,
                delete_tmp_folder_after_terminate=False,
                delete_output_folder_after_terminate=False,
                initial_configurations_via_metalearning=0,
                seed=i,
            )
        
        #Fit ensemble, change size and nbest if necessary
        cls.fit_ensemble(
                y_train,
                task=MULTICLASS_CLASSIFICATION,
                metric=accuracy,
            )

        anytime_model = cls


# In[130]:


#a = np.load('/tmp/option2_1_tmp/.auto-sklearn/ensembles/3.0000000000.ensemble', allow_pickle = True)
#print(a)


# In[ ]:




