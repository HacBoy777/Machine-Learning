import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class IrisDataset:
    
    def load_data(dir_path,queries=[],features=[],labels=[],test_size=0.25,scale=True):
        # print(dir_path)
        file_path = dir_path + '/Iris.csv'
        df= pd.read_csv(file_path)
        # print(df.head())
        df= df.fillna(0.0)
        # print(queries)
        
        for query in queries:
            df= df.query(query)
            # print(query)
            # print(df.head())
            
        X=df[features]                          ## FEATURES 
        # print(X.shape)    
        
        y=df[labels]                            ## LABELS
        # print(y.shape)
        print(type(y))
        
        mapping = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2
         }
        y = y.iloc[:, 0].map(mapping).values
        print(type(y))
        print(y[:3])
        y = pd.DataFrame(y)
        print(type(y))
        
        if scale:
            scaler = MinMaxScaler()
            X=scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return (X_train, y_train), (X_test, y_test)