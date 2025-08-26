import numpy as np
import matplotlib.pyplot as plt

def f(x, x0=1.2, depth=3.0, width=0.04):
    """
    x0:   location of the sharp drop (global minimum)
    depth:how deep the drop is (larger -> deeper)
    width:controls sharpness of the drop (smaller -> sharper)
    """
    x = np.asarray(x)
    well = -depth * np.exp(-((x - x0)**2) / (2*width**2))           # sharp, narrow well
    ripples = 0.25*np.sin(2.2*x) + 0.15*np.cos(5.3*x)               # local minima/maxima
    trend = 0.02*(x - x0)**4 + 0.02*x**2                            # keeps it bounded & smooth
    return well + ripples + trend

# Quick demo
x = np.linspace(-4, 4, 4000)
y = f(x)

plt.figure(figsize=(7,4))
plt.plot(x, y, lw=2)
plt.title("Sharp drop with nearby smooth terrain and local minima")
plt.xlabel("x"); plt.ylabel("f(x)"); plt.grid(True, alpha=0.3)
plt.show()
