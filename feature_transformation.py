# -*- coding: utf-8 -*-
"""
Feature transformation.
"""
import numpy as np

def transform_sigmoid(data,opt_list,opf_model,bias):
    #print("-------------------transform_sigmoid-------------------")
    from scipy.stats import logistic
    data_transformed = np.zeros((data.shape[0],len(opt_list.keys())))
    j = 0
    theta = 0.7#0.6#0.7#0.8#0.9
    for key in [*opt_list.keys()]:
        features = opt_list[key]
        W = bias - opf_model[features,1]        
        for i in range(0,data.shape[0]):
            X = data[i,features]
            s = logistic.cdf(np.inner(W,X))
            if s > theta :
                data_transformed[i,j] = 1
        #print("j = %d"%(j))
        j += 1
    return data_transformed    