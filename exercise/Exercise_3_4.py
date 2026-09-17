"""
Exercise 3.4
Modify the Python program from Example 3.6 so that it plots the histogram
of radius, size, texture, and smoothness of all the data points.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer(as_frame=True)
df = cancer.frame

# radius -> 'mean radius', size -> 'mean area', texture -> 'mean texture',
# smoothness -> 'mean smoothness'
df[['mean radius', 'mean area', 'mean texture', 'mean smoothness']].hist(figsize=(8, 6))
plt.tight_layout()
plt.show()


"""
Output aktual (hasil eksekusi nyata):

"""
