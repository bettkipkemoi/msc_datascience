import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
   
    # Load the image using Pillow
    image = Image.open(image_path)
    # Convert to grayscale
    grayscale_image = image.convert("L")
    # Normalize pixel values to range [0, 1]
    image_matrix = np.array(grayscale_image) / 255.0
    return image_matrix

def perform_svd(image_matrix):
   #TODO
   img = image_matrix
   U, S, Vt = np.linalg.svd(img)

def reconstruct_image(U, Sigma, Vt, k):
    #TODO
    k_values = 50
    reconstructed = np.dot(U[:, :k_values], np.dot(np.diag(S[:k_values]), Vt[:k_values, :]))
    
    return reconstructed

def visualize_reconstructions(image_matrix, U, Sigma, Vt, k_values):
   
    plt.figure(figsize=(12, 8))
    
    # Original image
    plt.subplot(2, len(k_values) + 1, 1)
    plt.imshow(image_matrix, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    # Reconstructed images
    for i, k in enumerate(k_values):
        reconstructed = reconstruct_image(U, Sigma, Vt, k)
        plt.subplot(2, len(k_values) + 1, i + 2)
        plt.imshow(reconstructed, cmap='gray')
        plt.title(f"k = {k}")
        plt.axis('off')
    
    # Plot singular values
    plt.subplot(2, len(k_values) + 1, len(k_values) + 2)
    plt.plot(Sigma, 'b-o')
    plt.title("Singular Values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # TODO: Load an image
    # Load an example image (provide the path to your image file)
    image_path = "/Users/bett/downloads/Module 5_ Activity 2/image.jpg"  # Change this to your image file path
    image_matrix = load_image(image_path)
    
    #TODO  Perform SVD
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    perform_svd(image_matrix)
    
    #TODO:  Visualize reconstructions
    k_values = [5, 10, 20, 50]  # Different values of k for reconstruction
    visualize_reconstructions(image_matrix, U, S, Vt, k_values)