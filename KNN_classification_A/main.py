from dataset.iris_dataset import IrisDataset
from sklearn.model_selection import train_test_split
import numpy as np
from model.Classification.knn_classification import KNN_Classification


IRIS_DATA_PATH="data"
queries=["Species == 'Iris-setosa' | Species == 'Iris-versicolor' |Species == 'Iris-virginica'"]
features=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
labels=["Species"]
def main():
    (X_train,X_test),(y_train,y_test)=IrisDataset.load_data(IRIS_DATA_PATH,queries,features,labels,scale=True)
    # print("shape of train",X_train.shape)(112,4)
    # print("shape of test",X_test.shape)(38,4)
    # print("shape of train",y_train.shape)(112,1)
    # print("shape of test",y_test.shape)(38,1)
    # k=3
    for k in range(1,10,2):
        print("K = ",k)
        model=KNN_Classification(n_neighbors=k)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        # print("Predicted labels:", y_pred)
        y_pred_train=model.predict(X_train)
        # y_pred_test=model.predict(X_test)
        print("Accuracy on Testing Data : ",model.accuracy(y_test,y_pred))
        print("Accuracy on Training Data : ",model.accuracy(y_train,y_pred_train))   
         
if __name__ == '__main__':
    main()