"""
Exercise 3.10
Modify the Python program from Example 3.14 so that it performs the
classification on Scikit-Learn's diabetes data
(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html).
"""
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["SVM", "Naive Bayes", "LDA",
         "QDA", "Decision Tree", "Random Forest",
         "Nearest Neighbors", "Neural Networks"]

classifiers = [
    SVC(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    MLPClassifier(alpha=1, max_iter=1000)]

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

for name, clf in zip(names, classifiers):
    try:
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(name + ": " + str(score))
    except ValueError as e:
        # Some classifiers (e.g. LDA/QDA) cannot handle classes that only
        # have 1 sample, which happens here because the diabetes target
        # is continuous (so almost every value is its own "class").
        print(name + ": FAILED - " + str(e))

# Note: same caveat as Exercise 3.9 - the diabetes target is continuous,
# so treating it as classification labels is not really appropriate; the
# very low scores below confirm this and illustrate the point.


"""
Output aktual (hasil eksekusi nyata):
SVM: 0.00904977375565611
Naive Bayes: 0.0
LDA: 0.004524886877828055
QDA: FAILED - y has only 1 sample in class 25.0, covariance is ill defined.
Decision Tree: 0.004524886877828055
Random Forest: 0.0
Nearest Neighbors: 0.0
Neural Networks: 0.0
"""
