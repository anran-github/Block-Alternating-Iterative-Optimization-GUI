import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc  # for LHS

plt.rcParams['font.family'] = 'Times New Roman'

# -------------------------------
# Define nonlinear bounded 2D function
# -------------------------------
def f(x, y, x0=1.2, depth=5.0, width=0.2):
    x = np.asarray(x)
    well = -depth * np.exp(-((x - x0)**2) / (2*width**2))           # sharp, narrow well
    ripples = 0           # local minima/maxima
    trend = 0.02*(x - x0)**4 + 0.02*(x - x0)**2                            # keeps it bounded & smooth
    
    return well + ripples + trend + 10 + 10*(y-x0)**2

# Domain
x_min, x_max = 0, 2.1
y_min, y_max = 0, 2.1

# Meshgrid for visualization
X = np.linspace(x_min, x_max, 200)
Y = np.linspace(y_min, y_max, 200)
XX, YY = np.meshgrid(X, Y)
ZZ = f(XX, YY)

# -------------------------------
# Sampling methods
# -------------------------------
n_samples = 40  # number of samples

# 1. Latin Hypercube Sampling (LHS)
sampler = qmc.LatinHypercube(d=2)
lhs_sample = sampler.random(n=n_samples)
lhs_points = qmc.scale(lhs_sample, [x_min, y_min], [x_max, y_max])

# 2. Grid Sampling
grid_x = np.linspace(x_min, x_max, int(np.sqrt(n_samples)))
grid_y = np.linspace(y_min, y_max, int(np.sqrt(n_samples)))
GX, GY = np.meshgrid(grid_x, grid_y)
grid_points = np.vstack([GX.ravel(), GY.ravel()]).T

# -------------------------------
# Plotting (Single Graph)
# -------------------------------
# plt.figure(figsize=(8, 6))
cs = plt.contourf(XX, YY, ZZ, levels=40, cmap="viridis")

# Plot both sampling methods
plt.scatter(lhs_points[:, 0], lhs_points[:, 1], 
            c="red", edgecolor="k", s=40, label="LHS")
plt.scatter(grid_points[:, 0], grid_points[:, 1], 
            c="blue", edgecolor="k", s=40, label="Grid Sampling")

plt.title("Color Map Representation of $f(x_1,x_2)$ Over the Input Domain")
plt.xlabel("$x_1$",fontsize=12)
plt.ylabel("$x_2$",fontsize=12)
plt.legend(loc="upper right")
plt.colorbar(cs, shrink=0.8)
plt.tight_layout()
plt.savefig("combined_sampling_2D.png", dpi=300)
plt.show()
