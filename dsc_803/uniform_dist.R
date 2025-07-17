#generating 1000 numbers from a uniform function ranging from 0 - 1
set.seed(123)  # For reproducibility
data <- runif(1000, min = 0, max = 1)
hist(data, main="Histogram of Uniform Distribution", xlab="Value", breaks=20)

