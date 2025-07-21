import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import image as mpimg

# Load a sample image (use a local image file or download one)
image = mpimg.imread('image.jpg')  # Ensure 'china.jpg' is in your working directory
image = image / 255.0  # Normalize pixel values to range [0, 1]
h, w, c = image.shape

# TODO Reshape the image for PCA (flatten the image)
image_flat = image.reshape(-1, c)
# TODO Apply PCA for image compression
n_components = 2  # You can adjust this value for more/less compression
pca = PCA(n_components=n_components)
image_pca = pca.fit_transform(image_flat)

# Reconstruct the image using the compressed PCA data
image_reconstructed = pca.inverse_transform(image_pca)
image_reconstructed = np.reshape(image_reconstructed, (h, w, c))

# Plot the original and compressed images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title('Reconstructed Image')
plt.imshow(image_reconstructed)
plt.show()