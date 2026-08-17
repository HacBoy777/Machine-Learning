# from dataset.iris_dataset import IrisDataset
# from models.clustering.kmeans_clustering import KmeansClustering
# import matplotlib.pyplot as plt


# IRIS_DATA_PATH="data"
# queries=["Species == 'Iris-setosa' | Species == 'Iris-versicolor' |Species == 'Iris-virginica'"]
# features=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
# labels=["Species"]

# def plot_clusters_centroids(X_train,clusters,centroids, X=0,y=1):
#     colors=["red","green","blue"]
#     for i in range(centroids.shape[0]):
#         data = X_train[clusters == i]
#         plt.scatter(data[:,X],data[:,y],color=colors[i],s=20)
#     plt.show()

# def main():
#     (X_train,X_test),(y_train,y_test)=IrisDataset.load_data(IRIS_DATA_PATH,queries,features,labels,scale=True)
#     k=3
#     model = KmeansClustering(k=k)
#     model.fit(X_train, y_train)
    
# if __name__ == '__main__':
#     main()

from dataset.iris_dataset import IrisDataset
from model.clustering.kmeans_clustering import KmeansClustering
import matplotlib.pyplot as plt
IRIS_DATA_PATH="data"
queries=["Species == 'Iris-setosa' | Species == 'Iris-versicolor' |Species == 'Iris-virginica'"]
features=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
labels=["Species"]
def plot_clusers_centoides(X_train,clusters,centroids,x=0,y=1):
    colors=['r','g','b','y']
    for i in range(centroids.shape[0]):
       data=X_train[clusters == i ]
       plt.scatter(data[:,x],data[:,y],c=colors[i],s=20)
    plt.scatter(centroids[:,x],centroids[:,y],marker="*",s=100,c="#012257")
    plt.show()
def main():
    (X_train,X_test),(y_train,y_test)=IrisDataset.load_data(IRIS_DATA_PATH,queries,features,labels,scale=True)
    k=3
    model=KmeansClustering(k=3)
    model.fit(X_train)
    plot_clusers_centoides(X_train,model.clusters,model.centroids,x=0,y=1)
    plot_clusers_centoides(X_train,model.clusters,model.centroids,x=2,y=3)
    plot_clusers_centoides(X_train,model.clusters,model.centroids,x=1,y=2)
if __name__ == '__main__':
    main()