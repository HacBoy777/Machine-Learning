from Dataset.iris_dataset import IrisDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from Model.LogisticRegModel import LogisiticRegression
from Visualization.myviz import myplot

IRIS_DIR_PATH="data"
queries=["Species == 'Iris-setosa' | Species == 'Iris-versicolor' "]
features=["SepalLengthCm","SepalWidthCm"]
labels=["Species"]
def main():
    (X_train,y_train),(X_test,y_test)=IrisDataset.load_data(IRIS_DIR_PATH,queries,features,labels,scale=True)
    # print("shape of train",X_train.shape):(75,2)
    # print("shape of test",X_test.shape):(25,2)
    # print("shape of train",y_train.shape):(75,1)
    # print("shape of test",y_test.shape):(25,1)
    # lr_model= LogisticRegression()
    # lr_model.fit(X_train,y_train)
    # pred_y=lr_model.predict(X_test)
    # print("predict of species",pred_y)
    # print(pred_y.shape)(25,)
    # print(y_train[:5])
    model=LogisiticRegression(learning_rate=0.3,max_iteration=100)
    losses=model.fit(X_train,y_train)
    # print(losses)
    y_pred_train=model.predict(X_train)
    # print(y_train[:5])
    # print(y_pred_train[:5])
    y_pred_test=model.predict(X_test)
    # print(y_test[:5])
    # print(y_pred_test[:5])
    # print("Accuracy on Testing Data : ",model.acccuracy(y_test,y_pred_test))
    # print("Accuracy on Testing Data : ",model.acccuracy(y_train,y_pred_train))
    ##visualise statr
    # myplot.plot(losses,"maxium iter","losses")
    # print(X_train.shape)(75,2)
    # print(y_train.shape)(75,1)
    myplot.plot_labelpoints(X_train,np.reshape(y_train,newshape=(y_train.shape[0])))
    myplot.plot_featurepoints(X_train,np.reshape(y_train,newshape=(y_train.shape[0])))
if __name__ == '__main__':
    main()