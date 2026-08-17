import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("Data\\insurance_data.csv")
# print(df.head())
# Training model
X = df[['age']]
y = df['bought_insurance']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
# print("X_train shape:", X_train.shape)  # 24,1
# print("X_test shape:", X_test.shape)  # 3,1
# print("y_train shape:", y_train.shape)  # 24,1
# print("y_test shape:", y_test.shape)  # 3,1

# age = int(input("Enter age: "))
# y_pred = model.predict([[age]])
# print(model.predict_proba([[age]]))
# print("Predicted values:", y_pred)
print(X_test)
print(model.predict(X_test))
print(model.predict_proba(X_test))


# plt.scatter(data.age, data.bought_insurance, marker='+', color='red')
# plt.xlabel("Age")
# plt.ylabel("Bought Insurance")
# plt.show()