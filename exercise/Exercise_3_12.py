"""
Exercise 3.12
Modify the Python program from Example 3.18 so that it performs multiple
linear regression on Scikit-Learn's Linnerud dataset; see:
https://scikit-learn.org/stable/datasets/toy_dataset.html#linnerrud-dataset
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_linnerud.html
"""
from sklearn import linear_model
from sklearn.datasets import load_linnerud

linnerud = load_linnerud()
# x: exercise data (Chins, Situps, Jumps)
# y: physiological data (Weight, Waist, Pulse) -- 3 targets at once
x = linnerud.data
y = linnerud.target

reg = linear_model.LinearRegression()
reg.fit(x, y)
print('Coefficients: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)

# Predict physiological data for someone who did 5 chin-ups, 162 sit-ups,
# and 60 jumps
pred = reg.predict([[5, 162, 60]])
print('Prediction: \n', pred)


"""
Output aktual (hasil eksekusi nyata):
Coefficients: 
 [[-0.47502636 -0.21771647  0.09308837]
 [-0.13687023 -0.04033662  0.0279736 ]
 [ 0.00107079  0.04202941 -0.02946117]]
Intercept: 
 [208.23351881  40.59787542  52.04362105]
Prediction: 
 [[176.17362115  35.05740701  57.09006881]]
"""
