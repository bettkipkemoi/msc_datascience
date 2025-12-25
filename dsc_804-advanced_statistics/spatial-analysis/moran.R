#load package
library(spdep)

# simulate dataset
n <- 10
coords <- cbind(runif(n, 0, 100), runif(n, 0, 100))
y <- sample(1:15, n, replace = TRUE)

# Create a Spatial Weights Matrix Define neighboring relationships using the -nearest neighbors method:
nb <- knn2nb(knearneigh(coords, k = 3))
wm <- nb2mat(nb, style = "W", zero.policy = TRUE)

# Compute Spatially Lagged Values Aggregate the values of neighboring polygons to compute spatially lagged values:
ms <- cbind(id = rep(1:n, each = n), y = rep(y, each = n), value = as.vector(wm * y))
ms <- ms[ms[, 3] > 0, ]
ams <- aggregate(ms[, 2:3], list(ms[, 1]), FUN = mean)
ams <- ams[, -1]
colnames(ams) <- c("y", "spatially lagged y")

# Visualize with Moran Scatter Plot The Moran scatter plot illustrates the relationship between observed values and spatially lagged values
plot(ams, pch = 20, col = "red", cex = 2,
     xlab = "Observed y", 
     ylab = "Spatially lagged y", 
     main = "Moran Scatter Plot")
reg <- lm(ams[, 2] ~ ams[, 1])
abline(reg, lwd = 2)
abline(h = mean(ams[, 2]), lty = 2)
abline(v = mean(y), lty = 2)

# Compute Moran’s I The slope of the regression line represents Moran’s I, quantifying the strength of spatial autocorrelation:
coefficients(reg)[2]