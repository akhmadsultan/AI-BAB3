"""
Exercise 3.20
Modify the Python program from Example 3.28 using a different dataset for
classification.
"""
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1)

clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)


"""
Output aktual (hasil eksekusi nyata):
                               Accuracy  ...  Time Taken
Model                                    ...            
GaussianNB                     1.000000  ...    0.010008
CalibratedClassifierCV         1.000000  ...    0.026455
RidgeClassifier                1.000000  ...    0.008916
NuSVC                          1.000000  ...    0.009783
LogisticRegression             1.000000  ...    0.011857
RidgeClassifierCV              1.000000  ...    0.009048
LinearDiscriminantAnalysis     1.000000  ...    0.010159
ExtraTreesClassifier           0.977778  ...    0.061070
... (dipotong agar ringkas) ...
DecisionTreeClassifier         0.955556  ...    0.011070
BaggingClassifier              0.933333  ...    0.026274
AdaBoostClassifier             0.933333  ...    0.064343
BernoulliNB                    0.933333  ...    0.010095
ExtraTreeClassifier            0.911111  ...    0.010522
DummyClassifier                0.377778  ...    0.009588

[26 rows x 7 columns]
"""
