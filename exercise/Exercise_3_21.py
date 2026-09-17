"""
Exercise 3.21
Modify the Python program from Example 3.28 using a different dataset for
regression.
"""
from sklearn.model_selection import train_test_split
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

reg = LazyRegressor()
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)

# Note: unlike Example 3.28d (which uses fetch_california_housing and
# needs to download data from the internet), sklearn's diabetes dataset
# is bundled locally, so this version works even without internet access.


"""
Output aktual (hasil eksekusi nyata):
                               Adjusted R-Squared  ...  Time Taken
Model                                              ...            
LinearRegression                         0.366436  ...    0.004183
TransformedTargetRegressor               0.366436  ...    0.004933
Ridge                                    0.363971  ...    0.004725
Lars                                     0.362032  ...    0.005344
LarsCV                                   0.360117  ...    0.009394
ElasticNetCV                             0.359855  ...    0.031847
LassoLarsIC                              0.359590  ...    0.005750
LassoLars                                0.359478  ...    0.004782
... (dipotong agar ringkas) ...
QuantileRegressor                       -0.136349  ...    0.011191
DecisionTreeRegressor                   -0.311839  ...    0.006760
ExtraTreeRegressor                      -0.381119  ...    0.005333
GaussianProcessRegressor                -1.024631  ...    0.012378
MLPRegressor                            -1.382569  ...    0.116627
KernelRidge                             -4.848846  ...    0.007558

[41 rows x 4 columns]
"""
