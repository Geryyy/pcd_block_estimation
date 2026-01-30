#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} mask.png")
    sys.exit(1)

path = sys.argv[1]

# Read mask *unchanged*
mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

if mask is None:
    raise RuntimeError(f"Could not read {path}")

print("dtype:", mask.dtype)
print("shape:", mask.shape)
print("min:", np.min(mask))
print("max:", np.max(mask))
print("unique (first 20):", np.unique(mask)[:20])

plt.figure(figsize=(6, 6))
plt.title("Mask heatmap")
plt.imshow(mask, cmap="hot")
plt.colorbar(label="pixel value")
plt.tight_layout()
plt.show()