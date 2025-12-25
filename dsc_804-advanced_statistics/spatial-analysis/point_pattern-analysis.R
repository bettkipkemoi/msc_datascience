# Load required libraries
library(spatstat)

# Create a non-regular polygonal boundary
coords <- matrix(c(
  0, 0,
  10, 0,
  8, 5,
  10, 10,
  5, 7,
  2, 10,
  0, 5,
  0, 0
), ncol = 2, byrow = TRUE)

# Define the polygon as an owin object
irregular_area <- owin(poly = list(x = coords[,1], y = coords[,2]))

# Simulate a random point pattern in the irregular area
set.seed(42)
nests <- rpoispp(lambda = 0.5, win = irregular_area)

# Perform density-based analysis (Kernel Density Estimation)
nest_density <- density(nests)
plot(nest_density, main = "Kernel Density of Bird Nests")

# Perform distance-based analysis (K-function)
k_result <- Kest(nests)
plot(k_result, main = "K-function for Bird Nests")
