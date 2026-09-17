"""
Exercise 3.15
Sketch the X data points on a piece of paper, add the third group of
points, and modify the Python program from Example 3.20 accordingly.
Make sure in each group only one point is labeled.

Note: to make the "one point labeled" plot easy to read, this solution
uses 2D points (x, y) instead of Example 3.20's 3D points, so the group
labels can be plotted clearly on a scatter plot.
"""
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 2], [1, 4], [1, 0],       # group 1
    [10, 2], [9, 4], [11, 0],     # group 2
    [5, 8], [4, 9], [6, 7],       # group 3 (new)
])

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
# Only ONE point per group is labeled, as requested
plt.annotate('Group 1', (X[0, 0], X[0, 1]))
plt.annotate('Group 2', (X[3, 0], X[3, 1]))
plt.annotate('Group 3', (X[6, 0], X[6, 1]))
plt.title('K-Means with 3 Groups (Exercise 3.15)')
plt.show()


"""
Output aktual (hasil eksekusi nyata):
[1 1 1 0 0 0 2 2 2]
[[10.  2.]
 [ 1.  2.]
 [ 5.  8.]]
"""
