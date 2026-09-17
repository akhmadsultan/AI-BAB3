"""
Exercise 3.9
Modify the Python program from Example 3.12 so that it performs random
forest classification on Scikit-Learn's diabetes data
(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html).
"""
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Total points: %d Correctly labeled points : %d" % (y_test.shape[0], (y_test == y_pred).sum()))

# Note: unlike Iris/Wine, the diabetes dataset's target ("y") is a
# CONTINUOUS disease-progression score (a regression target), not a class
# label. Feeding it into a *Classifier* means almost every sample gets its
# own unique "class", so accuracy will be very low/near zero - this is
# expected here and illustrates why choosing a classifier vs a regressor
# should match the type of the target variable.


"""
Output aktual (hasil eksekusi nyata):
Total points: 221 Correctly labeled points : 0
"""
