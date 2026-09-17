"""
Exercise 3.7
Modify the Python program from Example 3.10 so that it performs PCA on the
breast cancer data.
"""
import matplotlib.pyplot as plt
from sklearn import decomposition, datasets

# Load Breast Cancer data (instead of Iris)
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# Plot original data (first two features)
f = plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('mean radius')
plt.ylabel('mean texture')
plt.title('Original Breast Cancer Data')
f.show()

# Perform PCA
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X1 = pca.transform(X)

# Plot PCA data
g = plt.figure(2)
plt.scatter(X1[:, 0], X1[:, 1], c=y)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA of Breast Cancer Data')
plt.show()


"""
Output aktual (hasil eksekusi nyata):

"""
