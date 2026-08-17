# import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df= np.genfromtxt('Data\\data (1).csv', delimiter=',')
# df = pd.read_csv('Data/data (1).csv')
# print(df.head())
# print(df.shape)

X = df[:, [0]]
# print(X)
# print(X.shape)
y = df[:, 1]
# print(y)
# print(y.shape)

# plt.scatter(X, y, color ='red')
# plt.xlabel('Hours',color='blue')
# plt.ylabel('Marks',color = 'green')    
# plt.title('Hours of Study vs Marks Obtained',color='purple',fontsize=24) 
# plt.show()

## hyper parameters
learning_rate = 0.00008
max_itr = 1000

## Training Model

## Hypothesis 
def h(X,m,b):
    # print("m:",m)
    # print("b:",b)
    return m*X+b

def gradient(X,y,m,b):
    y_hat = h(X,m,b)
    # print(y_hat)
    dm = np.average((y_hat-y)*X)
    db = np.average(y_hat-y)
    # print("dm:",dm)
    # print("db:",db)
    return dm,db
    # return 0.,0.
    
def loss(X,y,m,b):
    y_hat = h(X,m,b)
    return np.average(np.square(y-y_hat))

def gradient_descent(X,y,learning_rate,max_itr):
    m = 0.
    b = 0.
    for i in range(max_itr):
        dm,db = gradient(X,y,m,b)
        m -=learning_rate*dm
        b -=learning_rate*db
        # print("m:",m)
        # print("b:",b)
        loss_value = loss(X,y,m,b)
        print("Loss:",loss_value)
    return m,b    
        

m,b = gradient_descent(X,y,learning_rate,max_itr)
print("Sloepe(m): ",m)
print("Intercept(b): ",b)