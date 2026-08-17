import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter
class KNN_Classification:
    def __init__(self,n_neighbors):
        self.n_neighbors=n_neighbors
        # print(self.n_neighbors)
        self.X=None
        self.y=None
    def fit (self,X,y):
        # print("X_train=",X.shape)(112,4)
        # print("y_train=",y.shape)(112,1)
        self.X=X
        self.y=y
    def predict(self,X_test):
        # print(X_test.shape)(38,4)
        y_pred=[]
        for test_X in X_test:##38 times
            dist_array=[]
            for train_X in self.X:##112 times
                dist=np.sum(np.square(test_X-train_X))
                dist_array.append(dist)
            # print("Length = ",len(dist_array))
            # print(dist_array)
            # list.sort(dist_array)
            # print(dist_array[:5])
            # print(dist_array[:self.n_neighbors])
            # print(self.y.values.shape)##(112,1)
            temp_y=self.y.values.reshape(self.y.values.shape[0])
            # print(temp_y.shape)##(112,)
            d,y=(list(t) for t in zip(*sorted(zip(dist_array,temp_y))))
            # print(d[:self.n_neighbors])
            # print(y[:self.n_neighbors])
            # print(y==temp_y)  
            y_labels=y[:self.n_neighbors]
            # print("ylabels :",y_labels)       
            b=Counter(y_labels)
            # print(b)
            class_name=(b.most_common(1)[0][0])
            # print(class_name)
            y_pred.append(class_name)
        return y_pred
    
    def accuracy(self,y_true,y_pred):
        return accuracy_score(y_true,y_pred)