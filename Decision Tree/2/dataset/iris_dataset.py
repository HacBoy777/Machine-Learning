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
            # print(qry)
            # print(df.head())
            
        # print(df)
        # print(df.shape)            ## (100, 6)
        
        X=df[features]
        y=df[labels].values
        # print(X.shape)             ## (150, 4)
        # print(y.shape)             ## (150,1)
        if scale:
            scaler = MinMaxScaler()
            X=scaler.fit_transform(X)
            
        # print(X[:5])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return (X_train,X_test),(y_train,y_test)