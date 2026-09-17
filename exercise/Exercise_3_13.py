"""
Exercise 3.13
Modify the Python program from Example 3.20 so that it uses Scikit-Learn's
function, sklearn.datasets.make_blobs(), to generate sample data points for
K-means clustering.
"""
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=0)

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.predict([[0, 0]]))


"""
Output aktual (hasil eksekusi nyata):
[2 1 2 0 0 0 2 2 0 2 1 1 1 2 1 0 2 2 0 1 0 2 1 2 0 0 2 0 1 1 0 2 2 1 1 0 1
 0 2 1 0 1 2 1 1 0 1 0 0 1 0 1 0 0 1 2 2 0 0 2 1 1 2 0 1 0 2 1 2 1 0 2 0 0
 1 2 1 0 2 2 0 2 1 2 2 2 1 0 2 2 0 1 0 2 1 1 2 1 0 2 1 0 2 1 2 2 0 2 1 1 2
 0 2 2 0 0 2 2 1 1 1 0 1 1 1 0 1 1 1 0 0 0 2 0 0 1 0 2 0 0 1 2 1 2 0 0 2 0
 0 1 2 0 2 1 0 0 1 1 2 1 2 2 1 2 0 2 2 2 2 0 1 2 0 1 1 1 2 1 2 2 1 0 2 2 2
 2 1 0 2 0 2 2 1 1 0 2 1 0 2 0 1 0 2 0 1 0 2 0 2 1 2 2 0 1 1 1 1 2 0 1 2 1
 1 1 2 0 0 2 2 2 2 1 1 2 1 0 0 0 2 2 1 0 0 0 0 1 2 0 2 1 1 2 1 1 0 2 1 0 2
 2 0 2 0 0 2 0 2 1 1 1 1 2 2 2 2 2 0 0 1 2 2 1 1 1 0 1 0 0 1 1 0 0 0 1 2 2
 0 1 2 2]
[[-1.72256762  2.8073623 ]
 [ 1.93204904  0.79614764]
 [ 0.90107601  4.33275275]]
[1]
"""
