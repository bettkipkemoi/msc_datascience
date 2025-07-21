import numpy as np
from sklearn.metrics import mean_squared_error

# Simulate a dataset of user ratings
np.random.seed(42)
num_users = 10
num_products = 8

# Create a ratings matrix with some missing values (represented by NaN)
ratings = np.random.randint(1, 6, size=(num_users, num_products)).astype(float)
mask = np.random.choice([0, 1], size=ratings.shape, p=[0.3, 0.7]).astype(bool)
ratings[~mask] = np.nan

print("Original Ratings Matrix:")
print(ratings)

# Low-rank matrix approximation using Singular Value Decomposition (SVD)
def low_rank_approximation(ratings, rank):
    # Replace NaN with the mean of each column for initialization
    #TODO
    # Replace NaN with column means
    filled_ratings = ratings.copy()
    col_means = np.nanmean(filled_ratings, axis=0)
    inds = np.where(np.isnan(filled_ratings))
    filled_ratings[inds] = np.take(col_means, inds[1])

    # Perform SVD
    U, S, Vt = np.linalg.svd(filled_ratings, full_matrices=False)
    # Keep only the top 'rank' singular values
    S_reduced = np.zeros_like(S)
    S_reduced[:rank] = S[:rank]
    S_matrix = np.diag(S_reduced)
    approx_ratings = U @ S_matrix @ Vt
    return approx_ratings

# Predict missing ratings
rank = 2
predicted_ratings = low_rank_approximation(ratings, rank)

print("\nPredicted Ratings Matrix:")
print(predicted_ratings)

# Analyze accuracy by comparing known values
#TODO
mse = mean_squared_error(ratings[~np.isnan(ratings)], predicted_ratings[~np.isnan(ratings)])

print("\nMean Squared Error of Predictions:", mse)

# Example recommendation for a specific user (e.g., user 0)
#TODO
user_id = 0  # You can change this to recommend for a different user

# Find products that the user has not rated (i.e., NaN in original ratings)
unrated_products = np.isnan(ratings[user_id])

# For these products, get the predicted ratings
user_predicted_ratings = predicted_ratings[user_id]

# Recommend top-N products (e.g., top 3) that the user hasn't rated yet
N = 3
# Get indices of unrated products sorted by predicted rating (descending)
recommended_indices = np.argsort(user_predicted_ratings[unrated_products])[::-1][:N]
# Map these indices back to the product indices in the full array
all_unrated_indices = np.where(unrated_products)[0]
recommended_products = all_unrated_indices[recommended_indices]

print("\nRecommended Products for User {}: ".format(user_id), recommended_products)

# Scalability analysis
print("\nScalability Analysis:")
print("Number of Users:", num_users)
print("Number of Products:", num_products)
print("Matrix Rank Used:", rank)
