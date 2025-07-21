"""
Goal: Use SVD for dimensionality reduction, specifically Principal Component Analysis (PCA).

Instructions:

Download a standard dataset such as the Iris dataset.
Perform SVD on the centered data matrix (after subtracting the mean from each feature):
Decompose the matrix into U, Σ, and V^T using SVD.
Use the top k singular values and corresponding singular vectors to project the data into -dimensional space.
Visualize the 2D projection of the Iris dataset using the top 2 principal components (i.e., the top 2 singular values).
Expected Outcome: A 2D scatter plot showing the Iris dataset projected onto its top two principal components, along with an explanation of how SVD is applied in PCA.
"""