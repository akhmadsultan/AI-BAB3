"""
Exercise 3.3
Modify the Python program from Example 3.4 so that it plots the first two
features (sepal length and sepal width) of all the data points as a
scatter plot.
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

species_codes, species_names = pd.factorize(df['species'])
plt.scatter(df['sepal_length'], df['sepal_width'], c=species_codes)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Sepal Width (Exercise 3.3)')
plt.show()


"""
Output aktual (hasil eksekusi nyata):

"""
