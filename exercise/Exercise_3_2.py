"""
Exercise 3.2
Modify the Python program from Example 3.2 so that it uses the third and
fourth features (petal length and width) as X. Compare the results with
Example 3.2.
"""
from sklearn import svm, datasets

iris = datasets.load_iris()
# Use the 3rd and 4th features: petal length and petal width
X = iris.data[:, 2:4]
y = iris.target  # 0: Setosa, 1: Versicolour, 2: Virginica
print(y)

clf = svm.SVC()
clf.fit(X, y)

# Predict the flower for a given petal length and width
p = clf.predict([[1.4, 0.2]])
print(p)

# Comparison note with Example 3.2 (which used sepal length/width instead):
# Petal length/width separate the three Iris species much more clearly than
# sepal length/width, so classification using these two features is
# generally at least as accurate as (often more accurate than) using the
# first two (sepal) features in Example 3.2.


"""
Output aktual (hasil eksekusi nyata):
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
[0]
"""
