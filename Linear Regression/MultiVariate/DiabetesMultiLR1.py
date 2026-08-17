import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Data\\diabetes.csv")
# print(df.head()) 
# print(df.shape) #(768, 9)
# print(df.columns)

# i = 1
# for col in df:
#     print(i, " . ", col)
#     i +=1
    
X = df.drop("Outcome", axis=1)
# print(X.shape)  #(768, 8)
y = df[["Outcome"]]
# print(y.shape)  #(768, 1)
# print(df[:5])

X = X - X.min()/ (X.max() - X.min())
# print(X[:5])
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# print(X[:5])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# print("X_train shape: ", X_train.shape)  # (614, 8)
# print("X_test shape: ", X_test.shape)    # (154, 8)
# print("y_train shape: ", y_train.shape)  # (614, 1)
# print("y_test shape: ", y_test.shape)    # (154, 1)

# Training the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
# print("Predicted values: \n", y_pred[:5])
# print(y_pred.shape)  # (154, 1)
# pred = lr_model.predict([[6,148,72,35,0,33.6,0.627,50]])
# print("Prediction for the input [6,148,72,35,0,33.6,0.627,50]: ", pred)
Preganancies = input("Enter Preganancies: ")
Preganancies = int(Preganancies)
Glucose = input("Enter Glucose: ")
Glucose = int(Glucose)
BloodPressure = input("Enter BloodPressure: ")
BloodPressure = int(BloodPressure)
SkinThickness = input("Enter SkinThickness: ")
SkinThickness = int(SkinThickness)
Insulin = input("Enter Insulin: ")
Insulin = int(Insulin)
BMI = input("Enter BMI: ")
BMI = float(BMI)
DiabetesPedigreeFunction = input("Enter DiabetesPedigreeFunction: ")
DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
Age = input("Enter Age: ")
Age = int(Age)
pred = lr_model.predict([[Preganancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
print("Prediction for the given input: ", pred)