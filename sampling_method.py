import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc  # for LHS

# -------------------------------
# Define nonlinear bounded function
# -------------------------------
def f(x, x0=1.2, depth=5.0, width=0.02):
    """
    x0:   location of the sharp drop (global minimum)
    depth:how deep the drop is (larger -> deeper)
    width:controls sharpness of the drop (smaller -> sharper)
    """
    x = np.asarray(x)
    well = -depth * np.exp(-((x - x0)**2) / (2*width**2))           # sharp, narrow well
    ripples = 0.25*np.sin(2.2*x) + 0.15*np.cos(5.3*x)               # local minima/maxima
    trend = 0.02*(x - x0)**4 + 0.02*x**2                            # keeps it bounded & smooth
    return well + ripples + trend + 10

# Domain
x_min, x_max = -2, 4
x_vals = np.linspace(x_min, x_max, 1000)
y_vals = f(x_vals)

# -------------------------------
# Sampling methods
# -------------------------------

# 1. Latin Hypercube Sampling (LHS)
sampler = qmc.LatinHypercube(d=1)
num_points = 10
lhs_sample = sampler.random(n=num_points//2)  # 30 points
lhs_points = qmc.scale(lhs_sample, x_min, x_max).flatten()

# 2. Grid Sampling
grid_points = np.linspace(x_min, x_max, num_points//2)

# 3. Monte Carlo (Random Uniform) Sampling
mc_points = np.random.uniform(x_min, x_max, num_points)

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
plt.scatter(np.concatenate([lhs_points,grid_points]), np.concatenate([lhs_vals,grid_vals]), c='red', marker='o', label="LHS Sampling")
# plt.scatter(grid_points, grid_vals, c='blue', marker='s', label="Grid Sampling")
plt.scatter(mc_points, mc_vals, c='green', marker='^', label="Monte Carlo Sampling")

plt.title("Sampling Methods Comparison")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("sampling_methods_1D.png", dpi=300)
plt.show()
