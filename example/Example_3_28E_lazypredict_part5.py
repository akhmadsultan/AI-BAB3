from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
diabetes = load_diabetes(as_frame=True)

X = diabetes.data
y = diabetes.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=1
)

# LazyPredict
reg = LazyRegressor()

models, predictions = reg.fit(
    X_train,
    X_test,
    y_train,
    y_test
)

# Tampilkan hasil
print(models)

# Plot R-Squared
plt.figure(figsize=(10, 5))

plt.plot(
    models.index,
    models["R-Squared"],
    "-s"
)

plt.xlabel("Model")
plt.ylabel("R-Squared")
plt.title("Perbandingan R-Squared Setiap Model")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()