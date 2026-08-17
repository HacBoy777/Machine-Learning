import numpy as np
from  dataset.iris_dataset import IrisDataset
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB

IRIS_DIR_PATH = "Data"
queries = ["Species == 'Iris-setosa' | Species == 'Iris-versicolor' | Species == 'Iris-virginica'"]
features=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
labels=["Species"]

def main():
    (X_train,y_train),(X_test,y_test)=IrisDataset.load_data(IRIS_DIR_PATH,queries,features,labels,scale=True)
    
    model= GaussianNB()
    model.fit(X_train,y_train)
    # print("TRAINING DATA :")
    y_pred_train=model.predict(X_train)
    # print(y_pred_train)
    # print("TESTING DATA : ")
    y_pred_test=model.predict(X_test)
    # print(y_pred_test)
    print("NAIVE BAYES -GaussianNB")
    print("Training Score :",model.score(X_train,y_pred_train))
    print("Testing Score :",model.score(X_test,y_pred_test))
    print("Training accuracy :",accuracy_score(y_train,y_pred_train))
    print("Testing accuracy :",accuracy_score(y_test,y_pred_test))
    
    model= MultinomialNB()
    model.fit(X_train,y_train)
    # print("TRAINING DATA :")
    y_pred_train=model.predict(X_train)
    # print(y_pred_train)
    # print("TESTING DATA : ")
    y_pred_test=model.predict(X_test)
    # print(y_pred_test)
    print("NAIVE BAYES -MultinomialNB")
    print("Training Score :",model.score(X_train,y_pred_train))
    print("Testing Score :",model.score(X_test,y_pred_test))
    print("Training accuracy :",accuracy_score(y_train,y_pred_train))
    print("Testing accuracy :",accuracy_score(y_test,y_pred_test))
    
    model= ComplementNB()
    model.fit(X_train,y_train)
    # print("TRAINING DATA :")
    y_pred_train=model.predict(X_train)
    # print(y_pred_train)
    # print("TESTING DATA : ")
    y_pred_test=model.predict(X_test)
    # print(y_pred_test)
    print("NAIVE BAYES -ComplementNB")
    print("Training Score :",model.score(X_train,y_pred_train))
    print("Testing Score :",model.score(X_test,y_pred_test))
    print("Training accuracy :",accuracy_score(y_train,y_pred_train))
    print("Testing accuracy :",accuracy_score(y_test,y_pred_test))
    
    model= BernoulliNB()
    model.fit(X_train,y_train)
    # print("TRAINING DATA :")
    y_pred_train=model.predict(X_train)
    # print(y_pred_train)
    # print("TESTING DATA : ")
    y_pred_test=model.predict(X_test)
    # print(y_pred_test)
    print("NAIVE BAYES -BernoulliNB")
    print("Training Score :",model.score(X_train,y_pred_train))
    print("Testing Score :",model.score(X_test,y_pred_test))
    print("Training accuracy :",accuracy_score(y_train,y_pred_train))
    print("Testing accuracy :",accuracy_score(y_test,y_pred_test))
    

if __name__ =='__main__':
    main()