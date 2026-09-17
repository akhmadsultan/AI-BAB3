# Example 3.27 - PyCaret Breast Cancer

# !pip install pycaret

import pandas as pd
from sklearn import datasets

# Load Breast Cancer dataset
cancer = datasets.load_breast_cancer(as_frame=True)

cancer.data['Target'] = cancer.target
cancer = cancer.data

print(cancer.head())

from pycaret import classification

classification.setup(
    data=cancer,
    target='Target'
)

classification.compare_models()