# -*- coding: utf-8 -*-
"""Quiz_2 -Calculus.ipynb

Original file is located at
    https://colab.research.google.com/drive/1bEEEhcQV--Vf3PPttEwgu0jXSqfwV9Jy

# Task 1: Write a dynamic function to find the derivative of any function f(x)
"""

#TO DO: Create a function that does a derivative for any function
from sympy import *
import matplotlib.pyplot as plt
import numpy as np
x = symbols('x')

def derivative(f):
    return diff(f, x)

# Example usage
f = x**2 + 2*x + 1
print(f"The derivative of {f} is {derivative(f)}")

"""# Task 2: Test the derivative function written with a quadratic equation of your choice
*NB: Must have atleast 2 minimas and atleast 2 maximas*
"""

x = symbols('x')

def derivative(f):
    return diff(f, x)

# Example usage
f = x**4 - 2*x**3 + 3*x**2 - 4*x + 5
print(f"The derivative of {f} is {derivative(f)}")

# Derivative of a quadratic equation with at least 2 minimas and 2 maximas
g = x**4 - 4*x**3 + 6*x**2 + 4*x + 1
print(f"The derivative of {g} is {derivative(g)}")

"""#Task 3: Plot a graph of the quadratic Equation"""

#Create a visualization of the quadratic equation
x = symbols('x')

def derivative(f):
    return diff(f, x)

# Define the quadratic equation
f = x**2 + 2*x + 1

# Define the range of x values
x_values = [i for i in range(-10, 11)]

# Calculate the y values for each x value
y_values = [f.subs(x, i) for i in x_values]

# Plot the graph
plt.plot(x_values, y_values)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Graph of Quadratic Equation of y={f}')
plt.show()

"""# Task 4: Create separate array of Minimas and maximas"""

from scipy.signal import argrelextrema

def find_minimas_maximas(f):
    x = np.linspace(-10, 10, 1000)
    y = f(x)
    minima_indices = argrelextrema(y, np.less)[0]
    maxima_indices = argrelextrema(y, np.greater)[0]
    minimas = x[minima_indices]
    maximas = x[maxima_indices]
    return minimas, maximas

# Example usage
f = lambda x: x**4 - 4*x**3 + 6*x**2 + 4*x + 1
minimas, maximas = find_minimas_maximas(f)
print(f"The minimas of {f} are {minimas}")
print(f"The maximas of {f} are {maximas}")

import sympy as sp

def get_maxima_minima(func_x):
    # Calculate the derivative
    x = sp.symbols('x')
    derivative = sp.diff(func_x, x)

    # Find critical points by solving f'(x) = 0
    critical_points = sp.solve(derivative, x)

    # Calculate y values for critical points
    x_critical_points = []
    y_critical_points = []
    for point in critical_points:
        y_value = func_x.subs(x, point)
        x_critical_points.append(point)
        y_critical_points.append(y_value)

    return x_critical_points, y_critical_points

# Example usage:
quadratic_eq = 4*x**2 + x**-1

x_val, y_val = get_maxima_minima(quadratic_eq)
# get the max and min
for x, y in zip(x_val, y_val):
  print("x =", x, ", y =", y)

"""#Finally, What is the global Minima and the Global maxima _ Plot this so that I can see"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def get_maxima_minima(func_x):
    """
    minimas and maximas of quadratic equation
    """
    x = sp.symbols('x')
    derivative = sp.diff(func_x, x)
    critical_points = sp.solve(derivative, x)
    x_critical_points = []
    y_critical_points = []
    for point in critical_points:
        y_value = func_x.subs(x, point)
        x_critical_points.append(point)
        y_critical_points.append(y_value)
    return x_critical_points, y_critical_points

def plot_maxima_minima(quadratic_eq):
    """
      Plot the maxima and minima of a quadratic equation.
    """
    x_values = np.linspace(-10, 100, 1000)
    y_values = sp.lambdify(x, quadratic_eq)(x_values)
    x_critical_points, y_critical_points = get_maxima_minima(quadratic_eq)
    plt.plot(x_values, y_values, label='Quadratic Equation')
    plt.plot(x_critical_points, y_critical_points, 'ro', label='G_Maxima/G_Minima')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Global Maxima and Minima of Quadratic Equation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example:
x = sp.symbols('x')
quadratic_eq = x**2

plot_maxima_minima(quadratic_eq)

"""# Things We did in Class

We came up with a functon

$$
f(x) = 4x^2 + x^-1
$$

Python Function



```
def f(x):
  #see our code below
```


"""

#This is an Example of a quadratic function
def f(x):
   return (4 * (x** 2)) + (x ** -1)

"""We manuall calculated the derivative  and got that
$$
df(x)/dx = 8x^1 + x^-2
$$

Using the formula above we find points where the derivative is 0

the values are:

$$
x1 = 0.5, x1 = - 0.5,x1 = 0,
$$

If you replace this in our original formula

$$
f(0.5) = 4(0.5)^2 + (0.5^-1 = -1.0
$$
$$
f(0) = 4x^2 + x^-1 = No solution
$$
$$
f(-0.5) = 4x^2 + x^-1 = 1
$$

Meaning our curve is flat at point $$(0.5,1) $$ and at $$ (-0.5,1)$$

Proof Pending........

Here is how we tried with code and got some errors
"""

x1 = 0.5
x2 = -0.5
x3 = 0

y1 = f(x1)
y2 = f(x2)
y3 = f(x3)

print(y2)
(0.5,3)

"""# **You can now do the Rest.All the Best ........**"""