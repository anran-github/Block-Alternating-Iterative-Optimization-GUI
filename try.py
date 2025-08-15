import sympy as sp
import numpy as np

# Define function and variable
x = sp.symbols('x', real=True)
f = 1000.0*(0.1*x - 1)**3 + 25.1639456178

# Constraints
a, b = -1, 2

# Second derivative
f2 = sp.diff(f, x, 2)
print(f2)
# Check convexity in constraint range
is_convex = all(f2.subs(x, val) >= 0 for val in np.linspace(a, b, 50))
print("Convex in constraints:", is_convex)
