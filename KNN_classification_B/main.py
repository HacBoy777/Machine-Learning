import numpy as np
from  dataset.iris_dataset import IrisDataset
from model.Classification.knn_classification import KNN_Classification
from sklearn.neighbors._regression import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

IRIS_DIR_PATH = "Data"
queries = ["Species == 'Iris-setosa' | Species == 'Iris-versicolor' | Species == 'Iris-virginica'"]
features=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
labels=["Species"]

def main():
    (X_train,y_train),(X_test,y_test)=IrisDataset.load_data(IRIS_DIR_PATH,queries,features,labels,scale=True)
    model= KNN_Classification(n_neighbours=5)
    print("My KNN Model")
    model.fit(X_train,y_train)
    print("Training Score ....")
    y_pred_train=model.predict(X_train)
    print("Testing Score ....")
    y_pred = model.predict(X_test)
    print("Accuracy on Training Data : ",model.accuracy(y_train,y_pred_train))   
    print("Accuracy on Testing Data : ",model.accuracy(y_test,y_pred))
    
    print("Sklearn KNN Model")
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train,y_train)
    print("Training Score ....")
    y_pred_train=model.predict(X_train)
    print("Testing Score ....")
    y_pred = model.predict(X_test)
    print("Model Score on Training Data : ",model.score(X_train,y_pred_train)) 
    print("Model Score on Testing Data : ",model.score(X_test,y_pred))
    
    print("Sklearn Linear Regression Model")
    model = LinearRegression()
    model.fit(X_train,y_train)
    print("Training Score :")
    y_pred_train = model.predict(X_train)
    print("Testing Score :")
    y_pred = model.predict(X_test)
    print("linear regression on Training Data : ",model.score(X_train,y_pred_train))
    print("linear regression on Testing Data : ",model.score(X_test,y_pred))
        
if __name__ =='__main__':
    main()