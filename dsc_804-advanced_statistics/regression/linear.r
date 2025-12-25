# Sample Data
data <- data.frame(x = c(1, 2, 3, 4, 5), y = c(3, 4, 2, 5, 6))

# Fit the model
model <- lm(y ~ x, data = data)

# Summary of the model
summary(model)