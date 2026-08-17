from dataset.iris_dataset import IrisDataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt
import seaborn as sn


IRIS_DATA_PATH="data"
queries=["Species == 'Iris-setosa' | Species == 'Iris-versicolor' |Species == 'Iris-virginica'"]
features=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
labels=["Species"]
def main():
    (X_train,X_test),(y_train,y_test)=IrisDataset.load_data(IRIS_DATA_PATH,queries,features,labels,scale=True)

    model = RandomForestClassifier(n_estimators=20)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    print('Model Score:', model.score(X_test, y_test))
    print("Actual labels:", y_test)
    print("Predicted labels:", y_pred)
    # conmat = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix:",conmat)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conmat, annot=True, cmap='Blues')
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.show()

if __name__ == '__main__':
    main()