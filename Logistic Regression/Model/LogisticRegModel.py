import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
class LogisiticRegression:
    def __init__(self,learning_rate,max_iteration):
        # print("hello")
        self.learning_rate=learning_rate
        self.max_iteration=max_iteration
        self.theta=None
        # print("Learning_rate",self.learning_rate)
        # print("Max iteration",self.max_iteration)
    def fit(self,X,y):
        losses=[]
        # print("X shape",X.shape)
        # print("y shape",y.shape)
        self.inilize_theta(X)
        for i in range(self.max_iteration):
            d_theta=self.cal_grad(X,y)
            self.theta=self.theta-self.learning_rate*d_theta
            loss=self.loss(X,y)
            losses.append(loss)
        return losses
    def loss(self,X,y):
        y_pred=self.pred_probability(X)
        # L = -[Y * log (Y) + (1-Y) log(1-Y)]
        return -np.average(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
    def inilize_theta(self,X):
        # print(X.shape)#(75,2)
        # print(X.shape[1])#(2)
        n_features=X.shape[1]+1
        # print(n_features)#(3)
        self.theta=np.zeros((X.shape[1]+1,1))
        # print(self.theta.shape)
        # print(self.theta)#(3,1)
    def cal_grad(self,X,y):
        # print(X.shape)
        # print(y.shape)
        # print(X[:5])
        y_pred=self.pred_probability(X)
        tempX=self.add_ones(X)
        d_theta=np.average((y_pred-y)*tempX,axis=0)
        # print("d_theta shape",d_theta.shape)##(3,)
        # print("d_theta",d_theta)
        d_theta=d_theta.reshape(d_theta.shape[0],1)
        # print(d_theta.shape)(3,1)
        ##d_theta=np.average((y_pred-y)*tempx,axis=0)
        # print(d_theta)
        return d_theta
    def pred_probability(self,X):
        tempX=self.add_ones(X)
        z=np.matmul(tempX,self.theta)
        # print("z shape",z.shape)##(75,1)
        # print(z)
        return self.sigmoid(z)
    def sigmoid(self,z):
        return np.exp(z)/(1+np.exp(z))
    def add_ones(self,X):
        tempX = np.ones(shape=(X.shape[0],X.shape[1]+1))
        # print("TempX shape",tempX.shape)##(75,3)
        # print(tempX)
        tempX[:,0:X.shape[1]]=X
        # print(tempX)
        return tempX  
    def predict(self,X,threshold=0.5):
        y_pred=self.pred_probability(X)
        y_hat=[]
        for y in y_pred:
            if y>=threshold:
                y_hat.append(1)
            else:
                y_hat.append(0)
        return np.array(y_hat)
    ###here v r converting list into numpy array 
    def acccuracy(self,y_true,y_pred):
        return accuracy_score(y_true,y_pred)