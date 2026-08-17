from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

## Data Preprocessing
digits=load_digits()
print(dir(digits)) 

# plt.gray() 
# for i in range(10):
#     plt.matshow(digits.images[i])
# plt.show()
# print(digits.DESCR)
# print(digits.data[5])

df =pd.DataFrame(digits.data)
# print(df.head())
# print(digits.target[5:10])

df['target']=digits.target
# print(df.shape)

X = df.drop('target',axis=1)
y = df[['target']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
# print("X_train:",X_train.shape) # 1437,64
# print("X_test:",X_test.shape)  # 360,64
# print("y_train:",y_train.shape)  # 1437,1
# print("y_test:",y_test.shape)  # 360,1

## Model Training
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
# print(model.score(X_test,y_test)) ## 0.9694444444444444
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# print("Train labels:\n", y_train_pred)
# print("Predicted labels:\n", y_test_pred)

conmat = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:\n", conmat)
plt.figure(figsize=(10,7))
sns.heatmap(conmat, annot=True, cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# print("Random Forest Score:")
# print("Train Score:", model.score(X_train, y_train_pred))
# print("Test Score:", model.score(X_test, y_test_pred))
# print("Random Forest Accuracy:")
# print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
# print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)
# print("Train labels:\n", y_train_pred)
# # print("Predicted labels:\n", y_test_pred)
# print("Decision Tree Score:")
# print("Train Score:", model.score(X_train, y_train_pred))
# print("Test Score:", model.score(X_test, y_test_pred))
# print("Decision Accuracy:")
# print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
# print("Test Accuracy:", accuracy_score(y_test, y_test_pred))