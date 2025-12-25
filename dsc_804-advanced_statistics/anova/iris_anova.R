# Load necessary libraries
library(ggplot2)
library(car)
library(dplyr)
library(DescTools)

# Step 1: Load the iris dataset
data(iris)

# Step 2: Perform one-way ANOVA
anova_result <- aov(Sepal.Length ~ Species, data = iris)

# Step 3: Check the summary of ANOVA
summary(anova_result)

# Step 4: Post-hoc analysis using Tukey's HSD test
post_hoc_result <- TukeyHSD(anova_result)
summary(post_hoc_result)

# Step 5: Check for assumptions

## Normality - Q-Q plot
par(mfrow = c(1, 3))  # Arrange the plots side by side
for (species in levels(iris$Species)) {
  qqnorm(iris$Sepal.Length[iris$Species == species], main = paste("Q-Q plot for", species))
  qqline(iris$Sepal.Length[iris$Species == species], col = "red")
}

## Homogeneity of variance - Levene's Test
leveneTest(Sepal.Length ~ Species, data = iris)

# Step 6: Interpretation of results

## ANOVA Results
cat("\nANOVA Results:\n")
anova_summary <- summary(anova_result)
print(anova_summary)

# Post-hoc analysis (Tukey's HSD test)
cat("\nPost-hoc Analysis (Tukey's HSD):\n")
print(post_hoc_result)

## Checking assumptions interpretation:
cat("\nAssumption Check Results:\n")
cat("1. Normality: Q-Q plots have been checked for each group.\n")
cat("2. Homogeneity of variance: Levene's Test p-value:\n")
