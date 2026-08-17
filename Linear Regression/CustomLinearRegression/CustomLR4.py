import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


data=np.genfromtxt("Data/data (1).csv",delimiter=',')
# print(data)
# print(data.shape)

X= data[:,[0]]
# print(X)
# print(X.shape)

y= data[:,1]
# print(y)
# print(y.shape)





## HYPER PARAMETER
learning_rate=0.0001
max_itr=100000

# Training Model
def h(X,m,b):
    # print("m= ",m)
    # print("b= ",b)
    return m*X+b
def gradient(X,y,m,b):
    y_hat=h(X,m,b)
    # print(y_hat)
    dm=np.average((y_hat-y)*X)
    db=np.average(y_hat-y)
    # print(dm,db)
    return dm,db
def loss(X,y,m,b):
    y_hat=h(X,m,b)
    return np.average(np.square(y-y_hat))
def gradient_descent(X,y,learning_rate,max_itr):
    m=0
    b=0
    losss=[]
    for i in range(max_itr):
        dm,db=gradient(X,y,m,b)
        m-=learning_rate*dm
        b-=learning_rate*db
        # print("m= ",m)
        # print("b= ",b)
        loss_value=loss(X,y,m,b)
        # print("loss_value =",loss_value)
        losss.append(loss_value)
    return m,b,losss,loss_value
# m,b,losss=gradient_descent(X,y,learning_rate,max_itr)
min_max_scaler=preprocessing.MinMaxScaler()
scaled_X=min_max_scaler.fit_transform(X)
# print(scaled_X)
# print(scaled_X.shape)
scaled_y=min_max_scaler.fit_transform(np.reshape(y,newshape=(y.shape[0],1)))#(y,newshape=(100,1))
# print(scaled_y)
# print(scaled_y.shape)


scaled_y=np.reshape(scaled_y,newshape=(scaled_y.shape[0]))
# print(scaled_y.shape)
m,b,losss,final_loss=gradient_descent(scaled_X,scaled_y,learning_rate,max_itr)

print("m=,b=",m,b)
print("final_loss",final_loss)

# print("losss",losss)
# plt.plot(losss)
# plt.title("Losss",fontsize=20)
# plt.xlabel("Number of ITR",fontsize=12)
# plt.ylabel("loss",fontsize=12)
# plt.show()
# plt.scatter(scaled_X, scaled_y,color="black")
# plt.xlabel("HOURS",color="green",fontsize="20")
# plt.ylabel("MARKS",color="red",fontsize="20")
# plt.title("LR for Expected Student Data : Expected marks vs Hours of study")
# y_pred=h(scaled_X,m,b)
# plt.plot(scaled_X,y_pred,c='r')
# plt.show()