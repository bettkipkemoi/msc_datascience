# Load necessary libraries
library(ggplot2)
library(car)
library(ggpubr)

# Load the mtcars dataset
data(mtcars)

# Convert 'cyl' (number of cylinders) and 'gear' (number of gears) into factors
mtcars$cyl <- as.factor(mtcars$cyl)
mtcars$gear <- as.factor(mtcars$gear)

# Perform Two-Way ANOVA
anova_model <- aov(mpg ~ cyl * gear, data = mtcars)

# Summary of the ANOVA results
summary(anova_model)

# Post-hoc analysis using Tukey's HSD test
tukey_results <- TukeyHSD(anova_model)
summary(tukey_results)

# Assumption Check: Normality of residuals
# Plotting residuals
par(mfrow = c(1, 2))
plot(anova_model, 1)  # Residuals vs Fitted plot
plot(anova_model, 2)  # Q-Q plot for residuals

# Levene's Test for Homogeneity of Variance
leveneTest(mpg ~ cyl * gear, data = mtcars)

# Reset plotting area
par(mfrow = c(1, 1))