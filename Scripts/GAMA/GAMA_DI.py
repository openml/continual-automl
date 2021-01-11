#!/usr/bin/env python
# coding: utf-8

# # GAMA Option-6  (Checkpoint)

# In[2]:


import arff
import numpy as np
import pandas as pd
import time

import sklearn.metrics
from gama import GamaClassifier
from gama.postprocessing import EnsemblePostProcessing
from gama.gama import Gama

from skmultiflow.data import FileStream
from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection import EDDM


# In[3]:


import sys
print(sys.argv[0]) # prints python_script.py
print(sys.argv[1]) # prints var1



# In[5]:


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
    if df.iloc[:,i].nunique()  > cat_limit:    #Limit for cardinality in categorical variables - 17 gave an error)
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


# In[15]:


#Overwrite search space by limited classifier space

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
    Binarizer,
)
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import (
    SelectFwe,
    SelectPercentile,
    f_classif,
    VarianceThreshold,
)

limited_config = {
    # Classifiers
    RandomForestClassifier: {
        "n_estimators": [100],
        "criterion": ["gini", "entropy"],
        "max_features": np.arange(0.05, 1.01, 0.05),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "bootstrap": [True, False],
        "warm_start":[True],
    },
    GradientBoostingClassifier: {
        "n_estimators": [100],
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "subsample": np.arange(0.05, 1.01, 0.05),
        "max_features": np.arange(0.05, 1.01, 0.05),
        "warm_start":[True],
    },
    Binarizer: {"threshold": np.arange(0.0, 1.01, 0.05)},
    FastICA: {"tol": np.arange(0.0, 1.01, 0.05)},
    FeatureAgglomeration: {
        "linkage": ["ward", "complete", "average"],
        "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"],
        "param_check": [
            lambda params: (not params["linkage"] == "ward")
            or params["affinity"] == "euclidean"
        ],
    },
    MaxAbsScaler: {},
    MinMaxScaler: {},
    Normalizer: {"norm": ["l1", "l2", "max"]},
    Nystroem: {
        "kernel": [
            "rbf",
            "cosine",
            "chi2",
            "laplacian",
            "polynomial",
            "poly",
            "linear",
            "additive_chi2",
            "sigmoid",
        ],
        "gamma": np.arange(0.0, 1.01, 0.05),
        "n_components": range(1, 11),
    },
    PCA: {"svd_solver": ["randomized"], "iterated_power": range(1, 11)},
    PolynomialFeatures: {
        "degree": [2],
        "include_bias": [False],
        "interaction_only": [False],
    },
    RBFSampler: {"gamma": np.arange(0.0, 1.01, 0.05)},
    RobustScaler: {},
    StandardScaler: {},
    # Selectors
    SelectFwe: {"alpha": np.arange(0, 0.05, 0.001), "score_func": {f_classif: None}},
    SelectPercentile: {"percentile": range(1, 100), "score_func": {f_classif: None}},
    VarianceThreshold: {"threshold": np.arange(0.05, 1.01, 0.05)},
}


# In[ ]:


from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

#Initialization
counter = 0

cls = GamaClassifier(max_total_time=3600, keep_analysis_log=None, n_jobs=1, scoring='log_loss', post_processing_method=EnsemblePostProcessing(),
                     config = limited_config)

drift_detector = EDDM()

start = 1
X_train = B[start-1].iloc[:,0:-1]
y_train = B[start-1].iloc[:,-1]


print("Starting to `fit`")
cls.fit(X_train, y_train)

anytime_model = cls

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
        drift_detector.add_element(int(not y_test.iloc[j] == y_hat[j]))
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
        
        #Refit the pipelines in the ensemble to the latest data. No reweighing or additional search.
        cls.model.fit(X_train, y_train)
        
        anytime_model = cls
        
    


# In[ ]:




