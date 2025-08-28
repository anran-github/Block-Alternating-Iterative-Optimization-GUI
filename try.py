import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load saved confusion matrix
cm = np.load("/home/anranli/Downloads/confunsion_matrix.npy")   # shape [n_classes, n_classes]

# Total number of pixels assessed
total_pixels = cm.sum()

# Row-normalized (recall view)
cm_row_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

# Column-normalized (precision view)
cm_col_norm = cm.astype(np.float64) / cm.sum(axis=0, keepdims=True)

# ================== Plot row-normalized ==================
disp = ConfusionMatrixDisplay(confusion_matrix=cm_row_norm,
                              display_labels=["Background", "Wing without Ice", "Green Dye", "White Ice"])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format=".2f", colorbar=False)
plt.title(f"Confusion Matrix for Recall Rate\nTotal pixels assessed = {total_pixels:,}")
plt.savefig("confusion_matrix_row_norm.png", dpi=300)
plt.show()

# ================== Plot column-normalized ==================
disp = ConfusionMatrixDisplay(confusion_matrix=cm_col_norm,
                              display_labels=["Background", "Wing without Ice", "Green Ice", "White Ice"])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap=plt.cm.Greens, values_format=".2f", colorbar=False)
plt.title(f"Confusion Matrix for Precision Rate\nTotal pixels assessed = {total_pixels:,}")
plt.savefig("confusion_matrix_col_norm.png", dpi=300)
plt.show()
