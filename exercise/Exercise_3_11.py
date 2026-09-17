"""
Exercise 3.11
Modify the Python program from Example 3.15 so that it has more data
points in x and y. Also, add an x label, y label, title, legend, and
grids to the plot.
"""
import matplotlib.pyplot as plt
from scipy import stats

# More data points than Example 3.15 (which only had 5)
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [3, 5, 5, 6, 7, 8, 7, 9, 10, 12]

slope, intercept, r, p, std_err = stats.linregress(x, y)


def myfunc(x):
    return slope * x + intercept


mymodel = list(map(myfunc, x))

plt.scatter(x, y, label='Data points')
plt.plot(x, mymodel, color='red', label=f'Fit line: y={slope:.2f}x+{intercept:.2f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression (Exercise 3.11)')
plt.legend()
plt.grid(True)
plt.show()

print("slope: ", slope)
print("intercept: ", intercept)


"""
Output aktual (hasil eksekusi nyata):
slope:  0.8484848484848485
intercept:  3.381818181818182
"""
