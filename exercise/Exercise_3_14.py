"""
Exercise 3.14
Sketch the X data points on a piece of paper, add two more points to each
group, and modify the Python program from Example 3.20 accordingly.
"""
from sklearn.cluster import KMeans
import numpy as np

# Example 3.20 had 3 points per group; here each group has 5 points
# (2 more added per group, staying close to the original group's location)
X = np.array([
    [1, 2, 3], [1, 4, 2], [1, 0, 3], [2, 3, 3], [0, 1, 2],       # group 1 (5 points)
    [10, 2, 4], [9, 4, 3], [11, 0, 2], [10, 3, 3], [9, 1, 4],    # group 2 (5 points)
])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.predict([[12, 3, 1]]))


"""
Output aktual (hasil eksekusi nyata):
[1 1 1 1 1 0 0 0 0 0]
[[9.8 2.  3.2]
 [1.  2.  2.6]]
[0]
"""
