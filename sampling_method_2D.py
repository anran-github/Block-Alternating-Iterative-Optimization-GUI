import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc  # for LHS

# -------------------------------
# Define nonlinear bounded 2D function
# -------------------------------
def f(x, y):
    return x**2 + 10*np.sin(5*x) + 5 * np.cos(10*x) +y**2 + 10*np.sin(5*y) + 5 * np.cos(10*y)

# Domain
x_min, x_max = 0,2.1
y_min, y_max = 0,2.1

# Meshgrid for visualization
X = np.linspace(x_min, x_max, 200)
Y = np.linspace(y_min, y_max, 200)
XX, YY = np.meshgrid(X, Y)
ZZ = f(XX, YY)

# -------------------------------
# Sampling methods
# -------------------------------

n_samples = 50  # number of samples

# 1. Latin Hypercube Sampling (LHS)
sampler = qmc.LatinHypercube(d=2)
lhs_sample = sampler.random(n=n_samples)
lhs_points = qmc.scale(lhs_sample, [x_min, y_min], [x_max, y_max])

# 2. Grid Sampling
grid_x = np.linspace(x_min, x_max, int(np.sqrt(n_samples)))
grid_y = np.linspace(y_min, y_max, int(np.sqrt(n_samples)))
GX, GY = np.meshgrid(grid_x, grid_y)
grid_points = np.vstack([GX.ravel(), GY.ravel()]).T

# 3. Monte Carlo (Random Uniform) Sampling
mc_points = np.random.uniform([x_min, y_min], [x_max, y_max], size=(n_samples, 2))

# -------------------------------
# Plotting
# -------------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Contour background
for ax, points, title, color in zip(
    axs,
    [lhs_points, grid_points, mc_points],
    ["LHS Sampling", "Grid Sampling", "Monte Carlo Sampling"],
    ["red", "blue", "green"]
):
    cs = ax.contourf(XX, YY, ZZ, levels=40, cmap="viridis")
    ax.scatter(points[:, 0], points[:, 1], c=color, edgecolor="k", s=40, label=title)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.colorbar(cs, ax=ax, shrink=0.8)

plt.suptitle("2D Function Sampling Comparison", fontsize=16)
plt.tight_layout()
plt.show()
