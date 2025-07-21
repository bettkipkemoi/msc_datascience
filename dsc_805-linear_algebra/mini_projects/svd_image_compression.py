import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Function to load an image
def load_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(image)

# Function to perform SVD and compress the image
def svd_compress(image, num_singular_values):
   #TODO
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    S_k = np.diag(S[:num_singular_values])
    image_reconstructed = U[:, :num_singular_values] @ S_k @ Vt[:num_singular_values, :]
    return image_reconstructed

# Function to plot original and compressed images
def plot_images(original, compressed, num_singular_values, compression_ratio):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
   #TODO
    ax[0].imshow(original, cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(compressed, cmap='gray')
    ax[1].set_title(f'Compressed Image (k={num_singular_values})')
    plt.show()

# Function to calculate and return the compression ratio
def calculate_compression_ratio(image, num_singular_values):
    #TODO
    compression_ratio = num_singular_values / (image.shape[0] * image.shape[1])
    return compression_ratio

# Function to evaluate the quality of the compressed image
def evaluate_quality(original, compressed):
    mse = np.mean((original - compressed) ** 2)  # Mean squared error
    psnr = 20 * np.log10(np.max(original) / np.sqrt(mse))  # Peak signal-to-noise ratio
    
    return mse, psnr

# Main function to run the compression and evaluation process
def main(image_path, num_singular_values_list):
    # Load the image
    original_image = load_image(image_path)

    # Iterate over the list of singular values to compress and evaluate
    for num_singular_values in num_singular_values_list:
        #TODO
        compressed_image = svd_compress(original_image, num_singular_values)
        compression_ratio = calculate_compression_ratio(original_image, num_singular_values)
        mse, psnr = evaluate_quality(original_image, compressed_image)

        # Print the evaluation results
        print(f"Number of Singular Values: {num_singular_values}")
        print(f"Compression Ratio: {compression_ratio:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"PSNR (Peak Signal-to-Noise Ratio): {psnr:.2f} dB")
        print("-" * 40)

        # Plot the original and compressed images
        plot_images(original_image, compressed_image, num_singular_values, compression_ratio)

if __name__ == "__main__":
    # Specify the image file path
    image_path = "image.jpg"  # Replace with your image file path
    
    # List of different singular values to evaluate the compression
    num_singular_values_list = [50, 100, 150, 200]  # Adjust as needed
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
    else:
        # Run the main function
        main(image_path, num_singular_values_list)