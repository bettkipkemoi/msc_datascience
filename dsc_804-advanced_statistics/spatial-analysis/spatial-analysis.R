# Load required libraries
library(terra)

# Define the URL for the shapefile (Natural Earth Admin 0 Countries)
url <- "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
output_zip <- "ne_countries.zip"
output_dir <- "natural_earth"

# Download the shapefile
download.file(url, output_zip, mode = "wb")

# Unzip the downloaded file
unzip(output_zip, exdir = output_dir)

# Load the shapefile
shapefile_path <- file.path(output_dir, "ne_110m_admin_0_countries.shp")
countries <- vect(shapefile_path)

# Inspect the data
print(countries)

# Plot the shapefile
plot(countries, col = "lightblue", border = "darkblue", main = "World Map (Natural Earth Data)")

# Compute adjacency matrix (Rook's case)
adj_matrix <- adjacent(countries, "rook", pairs = FALSE)

# View part of the adjacency matrix
print(adj_matrix)
print(adj_matrix[1:6, 1:6])

# Compute centroids of polygons
centroids <- centroids(countries)

# Compute distance-based neighbors (e.g., 100 km and 500 km thresholds)
dist_100 <- nearby(centroids, distance = 100000)  # 100 km
dist_500 <- nearby(centroids, distance = 500000)  # 500 km

# Compute k-nearest neighbors (e.g., 3 and 6 neighbors)
k3_neighbors <- nearby(centroids, k = 3)
k6_neighbors <- nearby(centroids, k = 6)

# Define a plotting function
plot_links <- function(nb, label) {
  plot(countries, col = "gray", border = "white")
  c1 <- centroids[nb[, 1], ]
  c2 <- centroids[nb[, 2], ]
  lines(c1, c2, col = "red", lwd = 2)
  title(label)
}

# Plot adjacency, distance-based, and nearest neighbor influences
par(mfrow = c(1, 1))
plot_links(nb = adjacent(countries, "rook"), label = "Adjacency (Rook's Case)")
plot_links(nb = dist_500, label = "Distance-based (500 km)")
plot_links(nb = k6_neighbors, label = "6-Nearest Neighbors")
