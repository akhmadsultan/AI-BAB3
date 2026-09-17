"""
Exercise 3.5
Modify the Python program from Example 3.7 so that it first trains the
naive Bayes model, saves the model to a file, loads the model from the
file, and makes a prediction with the model.
"""
import pickle
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)

# Train the model
clf = GaussianNB()
clf.fit(X, y)

# Save the trained model to a file
with open('naivebayes_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Load the model back from the file
with open('naivebayes_model.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)

# Make a prediction using the loaded model
p = loaded_clf.predict([[5.0, 3.4, 1.5, 0.4]])
print(p)


"""
Output aktual (hasil eksekusi nyata):
[0]
"""
