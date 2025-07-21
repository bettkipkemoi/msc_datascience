import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

img = color.rgb2gray(data.astronaut())
U, S, Vt = np.linalg.svd(img)

# Reconstruct using top k singular values
k = 50
reconstructed = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))

plt.imshow(reconstructed, cmap='gray')
plt.show()