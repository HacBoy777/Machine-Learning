import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

data=np.genfromtxt("Data/home.txt",delimiter=',')
# print(data)
# print(data.shape)  # (47, 3)

# X=data[:,0:2]
# y =data[:,2:]
# print(X.shape) # (47, 2)
# print(y.shape) # (47, 1)
# X1 = data[:,[0]]
# X2 = data[:,[1]]
# print(X1[:5])
# print(X2[:5])
# print(y[:5])

## Preprocessing
# min_max_scaler=preprocessing.MinMaxScaler()
# scaled_X=min_max_scaler.fit_transform(X)
# # print(scaled_X)
# print(scaled_X.shape)  # (47, 2)
# scaled_y=min_max_scaler.fit_transform(y)
# # print(scaled_y)
# print(scaled_y.shape)  # (47, 1)

## Hyper parameters

### Batch Gradient Descent
learning_rate=0.09
max_itr=500

### Stochistic Gradient Descent
s_learning_rate=0.06
s_max_itr=500

### mini-batch Gradient Descent
mb_learning_rate=0.09
mb_max_itr=500
mb_size=16

data = normalize(data,axis=0)
X=data[:,0:2]
y =data[:,2:]
# print(data[:5])
# print(data.shape)  # (47, 3)
# print(data.shape[0]) # 47
# print(data.shape[1]) # 3
# temp = np.zeros(data.shape[1])
# print(temp)  # [0. 0. 0.]
# print(temp.shape)  # (3,)
# tempX = np.ones(data.shape[1])
# print(tempX) # [1. 1. 1.]
# print(tempX.shape)  # (3,)
tempX = np.ones((X.shape[0], X.shape[1]+1))
# print(tempX[:5])
'''
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
'''
# print(tempX.shape)  # (47, 2)

# tempX = np.ones((X1.shape[0], 3))
# tempX[:, 1:2] = X1
# tempX[:, 2:3] = X2
# print(tempX[:5])
# print(tempX.shape)  # (47, 3)

tempX[:, 1:] = X
theta = np.zeros((X.shape[1]+1, 1))
s_theta = np.zeros((X.shape[1]+1, 1))
mb_theta = np.zeros((X.shape[1]+1,1))

# print(theta)
# print(theta.shape)  # (3, 1)
# print(tempX.shape)  # (47, 3)

# print(np.matmul(tempX,theta).shape)  # (47, 1)

def h(theta,X):
    tempX = np.ones((X.shape[0], X.shape[1]+1))
    tempX[:, 1:] = X
    return np.matmul(tempX,theta)

def gradient(theta,X,y):
    tempX = np.ones((X.shape[0], X.shape[1]+1))
    tempX[:,1:] = X
    d_theta = np.average((h(theta,X)-y)*tempX, axis=0)
    # print(d_theta.shape)
    d_theta = np.reshape(d_theta, newshape=(d_theta.shape[0],1))
    # print(d_theta.shape)
    return d_theta

def loss(theta,X,y):
    y_hat = h(theta,X)
    return np.average(np.square(y - y_hat))

# Batch Gradient Descent
print("\nBatch Gradient Descent:\n")
def gradient_decent(theta,X,y,learning_rate,max_itr,gap):
    cost = np.zeros(max_itr)
    
    for i in range(max_itr):
        d_theta = gradient(theta,X,y)
        theta = theta - learning_rate*d_theta
        # print(theta)
        cost[i] = loss(theta,X,y)
        if i%gap==0:
            print("Iteration:",i,"| loss:",loss(theta,X,y))
    return theta, cost

# # gradient(theta,X,y)
theta,cost = gradient_decent(theta,X,y,learning_rate,max_itr,100)
print("Final Theta[Batch Gradient]: ",theta)
# print("Final Loss:",loss(theta,X,y))
# print("Cost Array:",cost)

# Stochistic Gradient Descent
print("\nStochistic Gradient Descent:\n")
def stochistic_gradient_decent(s_theta,X,y,s_learning_rate,max_itr,gap):
    cost = np.zeros(s_max_itr)

    for i in range(s_max_itr):
        for j in range(X.shape[0]):
            d_theta = gradient(s_theta,X[j,:].reshape(1,X.shape[1]),y[j,:].reshape(1,y.shape[1]))
            # print(X[j,:].reshape(1,X.shape[1]).shape)  # (1, 2)
            s_theta = s_theta - s_learning_rate*d_theta     
            # print(y[j,:].reshape(1,1))  # (1, 1)
        cost[i] = loss(s_theta,X,y)
        if i%gap==0:
            print("Iteration:",i,"| loss:",loss(s_theta,X,y))
    return s_theta, cost
        
# Mini -Batch Gradient Descent
print("\nMini-Batch Gradient Descent:\n")
def mini_batch_gradient_decent(theta,X,y,learning_rate,max_itr,size,gap):
    cost = np.zeros(max_itr)
    for i in range(max_itr):
        for j in range(0,X.shape[0],size):
            d_theta = gradient(theta,X[j:j+size,:],y[j:j+size,:])
            theta = theta - learning_rate*d_theta
        cost[i] = loss(theta,X,y)
        if i%gap==0:
            print("Iteration:",i,"| loss:",loss(theta,X,y))
    return theta, cost

s_theta,s_cost = stochistic_gradient_decent(s_theta,X,y,s_learning_rate,s_max_itr,100)
print("Final Theta[Stochastic Gradient]:",s_theta)
# print("Final Loss:",loss(s_theta,X,y))
# print("Cost Array:",s_cost)

mb_theta,mb_cost = mini_batch_gradient_decent(mb_theta,X,y,mb_learning_rate,mb_max_itr,mb_size,100)
print("Final Theta[Mini-Batch Gradient]:",mb_theta)
# print("Final Loss:",loss(mb_theta,X,y))

# fig,ax = plt.subplots()
# ax.plot(np.arange(max_itr),cost,'r')
# ax.legend(loc = 'upper right',
#           labels = ['Batch Gradient Descent'])
# ax.set_xlabel('Iteration')
# ax.set_ylabel('Cost')
# ax.set_title('Error VS Training Epoch')
# plt.show()

fig,ax = plt.subplots()
ax.plot(np.arange(max_itr),cost,'b')
ax.plot(np.arange(s_max_itr),s_cost,'g')
ax.plot(np.arange(mb_max_itr),mb_cost,'r')
ax.legend(loc = 'upper right',
          labels = ['Batch Gradient Descent', 'Stochastic Gradient Descent', 'Mini-Batch Gradient Descent'])
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Error VS Training Epoch')
plt.show()