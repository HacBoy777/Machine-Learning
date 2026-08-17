import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
class IrisDataset:
     def load_data(dir_path,queries=[],features=[],labels=[],test_size=0.25,scale=True):
        file_path = dir_path + '/Iris.csv'
        df= pd.read_csv(file_path)
        df= df.fillna(0.0)
        for qry in queries:
            df= df.query(qry)
        
        X=df[features]
        y=df[labels]
        if scale:
            scaler = MinMaxScaler()
            X=scaler.fit_transform(X)
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return (X_train,X_test),(y_train,y_test)