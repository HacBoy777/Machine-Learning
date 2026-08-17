import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
from sklearn import linear_model      ## from sklearn import linear_model, datasets
from sklearn import datasets

diabetes=datasets.load_diabetes()
# print(diabetes.DESCR)       
# print(diabetes.keys())
# print(diabetes.data)          ## FEATURES
# print(diabetes.target)        ## LABEL
# print(diabetes.data.shape)      ## (442,10)

# First 10 Record
# print(diabetes.data[0])

# last 10 Record
# print(diabetes.data[441])

# Print first three records
# x = diabetes.data[:3]
# print(x.shape)

# print(diabetes.data.shape)   ## (442,10)
diabetes_X = diabetes.data[:, np.newaxis,9]  
# print(diabetes_X.shape)   ## (442,1,10)
print(diabetes_X[0:10])