import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('Data/income.csv')
# print(df.head())

scaler=MinMaxScaler()
df[['Age','Income($)']] = scaler.fit_transform(df[['Age','Income($)']])
# print(df[['Age','Income($)']][:5])

kmeans = KMeans(n_clusters=3)
y_pred = kmeans.fit_predict(df[['Age','Income($)']])
# print(y_pred)

df['Cluster'] = y_pred
# print(df)
# print(kmeans.cluster_centers_)

# plt.scatter(df['Age'],df['Income($)'])
# plt.xlabel('Age')
# plt.ylabel('Income($)') 
# plt.title('K-Means Clustering')
# plt.show()

# df1 = df[df['Cluster'] == 0]
# df2 = df[df['Cluster'] == 1]
# df3 = df[df['Cluster'] == 2]

# plt.scatter(df1['Age'],df1['Income($)'], color='red', label='Cluster 1')
# plt.scatter(df2['Age'],df2['Income($)'], color='blue', label='Cluster 2')
# plt.scatter(df3['Age'],df3['Income($)'], color='green', label='Cluster 3')
# plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color = 'yellow',label='Cluster Centers')
# plt.xlabel('Age')
# plt.ylabel('Income($)')
# plt.title('K-Means Clustering')
# plt.legend()
# plt.show()

## sse => sum of squared error
# print(kmeans.inertia_)  # 0.4750783498553096

sse = []
for i in range(1,10):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df[['Age','Income($)']])
    sse.append(kmeans.inertia_)
# print(sse)

k_range = range(1,10)
plt.plot(k_range, sse)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Sum of squared error')
plt.show()