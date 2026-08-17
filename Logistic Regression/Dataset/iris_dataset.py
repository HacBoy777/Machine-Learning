import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class IrisDataset:
    
    def load_data(dir_path,queries=[],features=[],labels=[],test_size=0.25,scale=True):
        # print(dir_path)
        file_path = dir_path + '/Iris.csv'
        df= pd.read_csv(file_path)
        # print(df)
        df= df.fillna(0.0)
        # print(queries)
        
        for qry in queries:
            df= df.query(qry)
            
        # print(df)
        # print(df.shape)            ## (100, 6)
        
        X=df[features]
        y=df[labels]
        # print(X.shape)             ## (100, 2)
        # print(y.shape)             ## (100,1)
        
        if scale:
            scaler = MinMaxScaler()
            X=scaler.fit_transform(X)
            
        # print(X[:5])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        new_y_train = np.zeros_like(y_train, dtype=float)
        for i in range(y_train.shape[0]):
            if y_train.iloc[i, 0] == "Iris-setosa":
                new_y_train[i, 0] = 1
            else:
                new_y_train[i, 0] = 0

        new_y_test = np.zeros_like(y_test, dtype=float)
        for i in range(y_test.shape[0]):
            if y_test.iloc[i, 0] == "Iris-setosa":
                new_y_test[i, 0] = 1
            else:
                new_y_test[i, 0] = 0
                
        # print(y_train[:5])
        # print(new_y_train[:5])
        # print(y_test[:5])
        # print(new_y_train[:5])

        return (X_train, new_y_train), (X_test, new_y_test)