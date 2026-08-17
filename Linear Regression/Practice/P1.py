import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Data/carprices.csv")
# print(df.head())
# print(df.shape) # (13, 4)

# plt.scatter(df['Mileage'],df['Sell Price($)'])
# plt.xlabel("Mileage")
# plt.ylabel("Sell Price($)")
# plt.show()

# plt.scatter(df['Age(yrs)'],df['Sell Price($)'])
# plt.xlabel("Age(yrs)")
# plt.ylabel("Sell Price($)")
# plt.show()

X = df[['Mileage','Age(yrs)']]
y = df[['Sell Price($)']]

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# print("X_train shape: ", X_train.shape)   # (9, 2)
# print("X_test shape: ",X_test.shape)  # (4, 2)
# print("y_train shape: ", y_train.shape)  # (9, 1)
# print("y_test shape: ", y_test.shape)  # (4, 1)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print("Predicted prices: ", y_pred)
