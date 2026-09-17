# Example 3.26 Auto_ml_test.py
from flaml import AutoML
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Mengambil dataset Boston Housing dari OpenML
boston = fetch_openml(
    name="boston",
    version=1,
    as_frame=True
)

df = boston.frame

# Memisahkan fitur dan target
X = df.drop(columns=["MEDV"])
y = df["MEDV"].astype(float)

# Membagi dataset menjadi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Membuat model AutoML untuk regresi
automl = AutoML()

# Melatih model
automl.fit(
    X_train,
    y_train,
    task="regression",
    time_budget=60
)

# Melakukan prediksi
y_pred = automl.predict(X_test)

# Menghitung evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model terbaik:", automl.best_estimator)
print("MSE:", mse)
print("R2 Score:", r2)
