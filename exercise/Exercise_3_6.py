"""
Exercise 3.6
Modify the Python program from Example 3.8 so that it uses 6 features and
2,000 samples. Compare the results with Example 3.8.
"""
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X, y = make_classification(n_samples=2000, n_features=6,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
print(X.shape)

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0, 0, 0]]))

# Comparison note with Example 3.8 (1000 samples, 4 features):
# With more samples (2000) and more features (6) but still only 2
# informative features, LDA has more data to estimate class covariances
# from, which generally makes the decision boundary more stable/reliable,
# though the extra non-informative features add noise the model must
# still separate out.


"""
Output aktual (hasil eksekusi nyata):
(2000, 6)
[0]
"""
