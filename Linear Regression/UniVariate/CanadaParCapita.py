# Write a program for liner regression predict in par capita income (per person inacome in country) of 
# year specify also plot graph with year on x axis and income in rupees on y axis 

import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
from sklearn import linear_model

df=pd.read_csv("data/canada_per_capita_income.csv")
# print("First five Records:", df.head())
print("Last five Records:", df.tail())
# print("Shape: ", df.shape)
X= df[["year"]]
y= df["per_capita_income"]
lr_model = linear_model.LinearRegression()

# Training Model
lr_model.fit(X,y)

## Testing Model
year = int(input("Enter Year to Predict Per Capita Income: "))
result = lr_model.predict([[year]])
print("Predicted Per Capita Income for {year} is : ", result)

# Plotting Graph
plt.scatter(X,y, color='blue')
plt.xlabel("Year",fontsize=20)
plt.ylabel("Per Capita Income (in Rs.)",fontsize=20)
plt.title("Per Capita Income Prediction", fontsize=25)
plt.show()