"""
Exercise 3.1
Modify the Python program from Example 3.1 to use six samples of X and y.
"""
from sklearn import svm

# Example 3.1 originally had 4 samples; here we use 6 (2 more added)
X = [[170, 70, 10], [180, 80, 12], [170, 65, 8], [160, 55, 7],
     [175, 72, 9], [155, 50, 6]]
# Height[cm], Weight[kg], Shoesize[UK]
y = [0, 0, 1, 1, 0, 1]  # Gender, 0: Male, 1: Female

clf = svm.SVC()
clf.fit(X, y)

# Predict
p = clf.predict([[160, 60, 7]])
print(p)


"""
Output aktual (hasil eksekusi nyata):
[1]
"""
