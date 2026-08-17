import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

class KNN_Classification:
    def __init__(self,n_neighbours):
        # print(n_neighbours)
        self.n_neighbours=n_neighbours
        # print("Knn : ",self.n_neighbours)
        self.X=None
        self.y=None
        
    def fit(self, X, y):
        # print("shape of X_train",X.shape)           ## (112, 4)
        # print("shape of y_train",y.shape)           ## (112, 1)
        self.X=X
        self.y=y
        
    def predict(self, X_test):
        y_pred = []
        for test_X in X_test:   # remove [:1]
            dist_array = []
            for train_X in self.X:
                dist = np.sum(np.square(test_X - train_X))
                dist_array.append(dist)
            temp_y = self.y.values.reshape(self.y.values.shape[0])
            d, y = (list(t) for t in zip(*sorted(zip(dist_array, temp_y))))
            y_labels = y[:self.n_neighbours]
            b = Counter(y_labels)
            class_name = b.most_common(1)[0][0]
            y_pred.append(class_name)
        return y_pred  
    
    def accuracy(self,y_true,y_pred):
        return accuracy_score(y_true,y_pred)