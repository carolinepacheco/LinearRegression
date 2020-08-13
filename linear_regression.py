# -*- coding: utf-8 -*-
"""
Nome: Caroline Pacheco do E. Silva
E-mail: lolyne.oacheco@gmail.com
"""

import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets


# Generate a random regression data
n_samples = 1000
n_outliers = 50


X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                      n_informative=1, noise=10,
                                      coef=True, random_state=0)

# The equation of a straight line is represented by: y = mx + b. 
# The (m, b) are two variables that can updat.
def hypothesis(m, b, x):
    return m + (b*x) 

# Display the graph ith the line 
def plot_line(m, b, X, y):
    max_x = np.max(X) + 100
    min_x = np.min(X) - 100

    xplot = np.linspace(min_x, max_x, 1000)
    yplot = m + b * xplot

    plt.plot(xplot, yplot, color='#ff0000', label='Regression Line')

    plt.scatter(X,y)
    plt.axis([-10, 10, 0, 200])
    plt.show()

# Initially, the values of a and b receive random values to begin with. 
m = np.random.rand()
b = np.random.rand()

# Display a line with randomly initialized m and b values.
plot_line(m, b, X, y)

# The line drawn in the graph above is wrong. 
# Therefore, we will calculate the total error of the line using the Mean Squared Error(MSE) function.

# Calculate the error function.
def cost(m, b, X, y):
    costValue = 0 
    for (xi, yi) in zip(X, y):
        costValue += 0.5 * ((hypothesis(m, b, xi) - yi)**2)
    return costValue

# Now we need to adjust the function to reduce this error.
# Calculate the derivatives using the following function.
def derivatives(m, b, X, y):
    dm = 0
    db = 0
    for (xi, yi) in zip(X, y):
        dm += hypothesis(m, b, xi) - yi
        db += (hypothesis(m, b, xi) - yi)*xi

    dm /= len(X)
    db /= len(X)

    return dm, db

# Using  the gradient update rule  to update the (m, b) parameters .    
def update_parameters(m, b, X, y, alpha):
    dm, db = derivatives(m, b, X, y)
    m = m - (alpha * dm)
    b = b - (alpha * db)

    return m, b


# Now we repeat these steps - checking the error, calculating the derivatives,
# and updating the weights until the error is as low as possible.
# This is called minimizing the cost function.

# Mnimizing the cost function    
def linear_regression(X, y):
    m = np.random.rand()
    b = np.random.rand()

    for i in range(0, 1000):
        if i % 100 == 0:
            plot_line(m, b, X, y)
        m, b = update_parameters(m, b, X, y, 0.005)


# call function
linear_regression(X, y)