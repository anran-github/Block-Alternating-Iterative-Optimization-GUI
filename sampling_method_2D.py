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
    ripples = 0
    trend = 0.02*(x - x0)**4 + 0.02*(x - x0)**2
    return well + ripples + trend + 10 + 10*(y - x0)**2

# -------------------------------
# Domain and mesh
# -------------------------------
x_min, x_max = 0, 2.1
y_min, y_max = 0, 2.1
X = np.linspace(x_min, x_max, 400)
Y = np.linspace(y_min, y_max, 400)
XX, YY = np.meshgrid(X, Y)
ZZ = f(XX, YY)

# Constraint mask
with np.errstate(divide='ignore', invalid='ignore'):
    feasible_mask = (XX >= 0) & (YY >= 0) & (XX > 0.5) & (YY <= 1 / (XX - 0.5))
    infeasible_mask = ~feasible_mask

# -------------------------------
# Sampling
# -------------------------------
n_samples = 40

# Latin Hypercube Sampling
sampler = qmc.LatinHypercube(d=2)
lhs_sample = sampler.random(n=n_samples)
lhs_points = qmc.scale(lhs_sample, [x_min, y_min], [x_max, y_max])

# Grid Sampling
grid_x = np.linspace(x_min, x_max, int(np.sqrt(n_samples)))
grid_y = np.linspace(y_min, y_max, int(np.sqrt(n_samples)))
GX, GY = np.meshgrid(grid_x, grid_y)
grid_points = np.vstack([GX.ravel(), GY.ravel()]).T

# Combine samples
all_points = np.vstack([lhs_points, grid_points])

# -------------------------------
# Feasibility filter & top 4
# -------------------------------
def is_feasible(x1, x2):
    return (x1 > 0.5) and (x2 >= 0) and (x2 <= 1 / (x1 - 0.5))

feasible_points = np.array([pt for pt in all_points if is_feasible(pt[0], pt[1])])
feasible_values = np.array([f(pt[0], pt[1]) for pt in feasible_points])

if len(feasible_values) >= 4:
    top4_idx = np.argsort(feasible_values)[:4]
    top4_points = feasible_points[top4_idx]
else:
    top4_points = feasible_points

# Separate feasible subsets for plotting
lhs_feasible = np.array([pt for pt in lhs_points if is_feasible(pt[0], pt[1])])
grid_feasible = np.array([pt for pt in grid_points if is_feasible(pt[0], pt[1])])

# -------------------------------
# Plotting
# -------------------------------
plt.figure(figsize=(8, 6))

# Main colormap
cs = plt.contourf(XX, YY, ZZ, levels=40, cmap="viridis")

# Infeasible region shaded red
plt.contourf(XX, YY, infeasible_mask, levels=[0.5, 1], colors='red', alpha=0.3)

# Constraint boundaries
plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label="Constraint Boundaries")
x1_vals = np.linspace(0.5 + 1e-3, x_max, 500)
x2_boundary = 1 / (x1_vals - 0.5)
plt.plot(x1_vals, x2_boundary, 'k--', linewidth=2)

# Sample points (only feasible ones)
if len(lhs_feasible) > 0:
    plt.scatter(lhs_feasible[:, 0], lhs_feasible[:, 1],
                c="red", edgecolor="k", s=60, label="LHS")
if len(grid_feasible) > 0:
    plt.scatter(grid_feasible[:, 0], grid_feasible[:, 1],
                c="blue", edgecolor="k", s=60, label="Grid Sampling")

# Top 4 selected feasible points
if len(top4_points) > 0:
    plt.scatter(top4_points[:, 0], top4_points[:, 1],
                c="gold", s=150, edgecolor="black", marker='*', label="Proposed Hybrid Method")

# Formatting
plt.title("Color Map Representation of $f(x_1,x_2)$ Over the Input Domain", fontsize=18)
plt.xlabel("$x_1$", fontsize=20)
plt.ylabel("$x_2$", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc="upper right", fontsize=14)

cbar = plt.colorbar(cs, shrink=0.8)
cbar.set_label("$f(x_1,x_2)$", fontsize=16)
cbar.ax.tick_params(labelsize=16)

plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('paper_plot/combined_sampling_2D.png', dpi=500)
plt.show()
