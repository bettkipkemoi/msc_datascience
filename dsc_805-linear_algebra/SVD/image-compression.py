"""
Goal: Apply SVD to image compression by reducing the rank of the matrix representation of an image.

Instructions:

Load a grayscale image into Python using libraries such as matplotlib.image or PIL.
Treat the image as a matrix  of pixel intensities and perform SVD:
A = UΣV^T
Reconstruct the image using only the top k singular values, where k is less than the full rank of the matrix. Test for different values of k, such as k=5, 20, 50, etc.
Plot the original image and the compressed versions using matplotlib.
Expected Outcome: Original and compressed versions of the image displayed side by side, with a discussion on the trade-off between image quality and the number of singular values used.
"""