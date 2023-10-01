# 1.
# Hien thi du lieu
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(x, y)
plt.show()

# su dung phuong phap elbow de tim gia tri k phu hop


data = list(zip(x, y))
inertias = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)


plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# gom nhom voi giai thuat kmeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()

# 2.


x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)

plt.show()


data = pd.read_csv('BTH5/USArrests.csv', delimiter=',')
print(data)
df = pd.DataFrame(
    scale(data.iloc[:, 1:]), index=data.index, columns=data.iloc[:, 1:].columns)
print(df)
# Khởi tạo đối tượng PCA với số comp = 2
my_pca = PCA(n_components=2)

# Fit vào data
my_pca.fit(df)

# Thực hiện transform
#df_2_dim = my_pca.transform(df)

df_2_dim = pd.DataFrame(my_pca.transform(df), index=df.index)
df_2_dim.head()
# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42).fit(df)

# Add cluster labels to dataframe
df['Cluster'] = kmeans.labels_

# Plot the cluster visualization
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel('Murder')
plt.ylabel('Assault')
plt.show()
