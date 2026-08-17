## Data science ~ pre processing
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

df=pd.read_csv("data\homeprices.csv")

## DATA SCIENCE

## MACHINE LEARNING
from sklearn import linear_model

print("Home Prediction Model, for Single Area Prediction")
lr_model= linear_model.LinearRegression()
x=df[["area"]]
y=df["price"]

## Training Model
lr_model.fit(x,y)

## Testing Model
df_test = pd.read_csv("data\\areas.csv")
# print(df_test)
pred_price = lr_model.predict(df_test)
df_test["pred_price"]= pred_price
print(df_test)
df_test
df_test.to_csv("data/predicted_prices.csv", index=False)      ## unknown csv to located csv use (..).to_csv
df_pred= pd.read_csv("data/predicted_prices.csv")

## Plotting Graph
## Title ==> Home Price Prdiction 
## X-axis ==> area(sq.feet)
## Y-axis ==> price(Rs.) 

              
plt.xlabel("area(sq.feet)",fontsize=20)
plt.ylabel("price(Rs.)",fontsize=20)
plt.title("Plotting Traning Data ")
plt.scatter(df.area, df.price,color='red',marker='+')
plt.plot(df.area, df.price,color='red',marker='+')  
plt.show()

plt.xlabel("area(sq.feet)",fontsize=20)
plt.ylabel("price(Rs.)",fontsize=20)
plt.title("Plotting Traning Data ")
plt.scatter(df_pred.area,df_pred.pred_price,color='red',marker='+')
plt.plot(df_pred.area,df_pred.pred_price,color='red',marker='+')
plt.show()