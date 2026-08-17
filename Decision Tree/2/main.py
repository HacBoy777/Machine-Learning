from dataset.iris_dataset import IrisDataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import graphviz

IRIS_DATA_PATH="data"
queries=["Species == 'Iris-setosa' | Species == 'Iris-versicolor' |Species == 'Iris-virginica'"]
features=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
labels=["Species"]
def main():
    (X_train,X_test),(y_train,y_test)=IrisDataset.load_data(IRIS_DATA_PATH,queries,features,labels,scale=True)
    # print("shape of train",X_train.shape)  #(112,4)
    # print("shape of test",X_test.shape)  #(38,4)
    # print("shape of train",y_train.shape)  #(112,1)
    # print("shape of test",y_test.shape)  #(38,1) 

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    # print("Train labels:\n", y_train_pred)
    # # print("Predicted labels:\n", y_test_pred)
    # print("Decision Tree Score:")
    # print("Train Score:", model.score(X_train, y_train_pred))
    # print("Test Score:", model.score(X_test, y_test_pred))
    # print("Decision Accuracy:")
    # print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    # print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    dot_data = tree.export_graphviz(model, out_file=None, feature_names=features, class_names=labels)
    graph = graphviz.Source(dot_data)
    graph.render("iris_decision_tree")
         
if __name__ == '__main__':
    main()