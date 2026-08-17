import matplotlib.pyplot as plt
class myplot:
    def plot(data=[],x_label="",y_label=""):
        # print("Plotting start")
        # print(data)
        # print(x_label)
        # print(y_label)
        plt.plot(data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    def plot_labelpoints(X_train,y_train):
        # print(X_train.shape)(75,2)
        # print(y_train.shape)(75,)
        plt.scatter(X_train[:,0],y_train)
        plt.show()
        plt.scatter(X_train[:,1],y_train)
        plt.show()
        
    def plot_featurepoints(X_train,y_train,x=0,y=1):
        colors=["r","g"]
        for i in range(2):##for tow binary class(0=>Iris-versicolor,1=>Iris-setosa)
            # print(i)
            # print(y_train==i)
            data=X_train[y_train == i ]
            plt.scatter(data[:,x],data[:,y],c=colors[i],s=50)
        plt.show()
        
