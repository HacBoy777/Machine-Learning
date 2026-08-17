import pandas as pd
import numpy as np
from word2number import w2n
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('Data/hiring.csv')
# print(df.head())
# print(df.shape)

df['experience'].fillna("Zero",inplace = True)
# # print(df)

df.experience = df.experience.apply(w2n.word_to_num)
# print(df.experience)

# df['test_score(out of 10)'] = df['test_score(out of 10)'].replace(0, np.nan)
# print(df)

# df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean(),inplace = True)
# print(df)

floor_mean = math.floor(df["test_score(out of 10)"].mean())
df["test_score(out of 10)"].fillna(floor_mean, inplace=True)
# print(df)

X = df[["experience", "test_score(out of 10)", "interview_score(out of 10)"]]
y = df["salary($)"]
experience = int(input("Enter experience: "))
test_score = float(input("Enter test score (out of 10): "))
interview_score = float(input("Enter interview score (out of 10): "))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train,y_train)
# y_pred = lr_model.predict(X_test)
# print("Predicted salaries: ", y_pred)
pred = lr_model.predict([[experience, test_score, interview_score]])
print("Predicted salary: ", pred)
