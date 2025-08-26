import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc  # for LHS

# -------------------------------
# Define nonlinear bounded function
# -------------------------------
def f(x):
    return x + 0.5 * np.sin(5*x) + 0.2 * np.cos(10*x)

# Domain
x_min, x_max = -5, 5
x_vals = np.linspace(x_min, x_max, 1000)
y_vals = f(x_vals)

# -------------------------------
# Sampling methods
# -------------------------------

# 1. Latin Hypercube Sampling (LHS)
sampler = qmc.LatinHypercube(d=1)
lhs_sample = sampler.random(n=30)  # 30 points
lhs_points = qmc.scale(lhs_sample, x_min, x_max).flatten()

# 2. Grid Sampling
grid_points = np.linspace(x_min, x_max, 30)

# 3. Monte Carlo (Random Uniform) Sampling
mc_points = np.random.uniform(x_min, x_max, 30)

# Evaluate function at sampled points
lhs_vals = f(lhs_points)
grid_vals = f(grid_points)
mc_vals = f(mc_points)

# -------------------------------
# Plotting
# -------------------------------
plt.figure(figsize=(12, 7))
plt.plot(x_vals, y_vals, 'k-', linewidth=2, label="Function f(x)")

# Plot sample points
plt.scatter(lhs_points, lhs_vals, c='red', marker='o', label="LHS Sampling")
plt.scatter(grid_points, grid_vals, c='blue', marker='s', label="Grid Sampling")
plt.scatter(mc_points, mc_vals, c='green', marker='^', label="Monte Carlo Sampling")

plt.title("Function with Unique Global Minimum + Local Minima\nSampling Comparison")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
