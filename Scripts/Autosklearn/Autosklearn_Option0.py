#!/usr/bin/env python
# coding: utf-8

# # Auto-sklearn Option 6 (Checkpointing)

# In[ ]:


import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import arff
import numpy as np
import pandas as pd
import time

from skmultiflow.data import FileStream
from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection import EDDM


# In[ ]:


import sys
print(sys.argv[0]) # prints python_script.py
print(sys.argv[1]) # prints var1


# In[9]:


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
    
    #columns = [alldata["attributes"][i][0] for i in range(0,len(alldata["data"][0]))]

if pd.isnull(df.iloc[:,:]).any().any():
    print("Data X contains NaN values")
    
df.dropna()

#Remove high cardinality categorical variables because of the bug in H2O
to_remove=[]

for i in categorical_columns:
    if df.iloc[:,i].max()  > cat_limit:    #Limit for cardinality in categorical variables - 17 gave an error)
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


# In[13]:


#Initialization

#Initial search space with all classifiers
cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600, 
                                                       include_estimators = ["random_forest", "gradient_boosting"])

#drift_detector = ADWIN()
drift_detector = EDDM()

X_train = B[0].iloc[:,0:-1]
y_train = B[0].iloc[:,-1]
start = 1


cls.fit(X_train, y_train)
anytime_model = cls
#models = [cls.get_models_with_weights()[i][1] for i in range(0,len(cls.get_models_with_weights()))]
#print(cls.get_models_with_weights())
                  
#Prequential evaluation

for i in range(start,n):
    
    #Test on next batch for accuracy
    X_test = B[i].iloc[:,0:-1]
    y_test = B[i].iloc[:,-1]
    y_hat = cls.predict(X_test)
    
    b_acc = sklearn.metrics.balanced_accuracy_score(y_test, y_hat)  #equivalent to ROC_AUC in binary case
    acc = sklearn.metrics.accuracy_score(y_test, y_hat)
    print("Test batch %d - Balanced accuracy %f - Accuracy %f\n" % (i, b_acc, acc))
    
    #Check for drift
    drift_in_batch = 0
    for j in range(0,len(B[i])):
        drift_detector.add_element(y_test.iloc[j] - y_hat[j])
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
            
        cls.refit(X_train, y_train)
    


# In[19]:


cls.show_models()


# In[ ]:


cls.get_models_with_weights()

