# Load ggplot2
library(ggplot2)

# Create a basic scatter plot
ggplot(data = mtcars, aes(x = wt, y = mpg)) +
  geom_point()
#aes(x = wt, y = mpg) defines the aesthetics of the plot, mapping the wt variable to the x-axis and the mpg variable to the y-axis. 
#geom_point() adds points to the plot, creating a scatter plot.

#customize your plots by adding layers, changing themes, and modifying scales.
# Create a scatter plot
ggplot(data = mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  labs(
    title = "Scatter Plot of Weight vs. Miles per Gallon",
    x = "Weight (1000 lbs)",
    y = "Miles per Gallon"
  ) +
  theme_minimal()

## plotly
#Install the package
install.packages("plotly")

# Load plotly
library(plotly)

# Basic scatter plot
plot_ly(data = mtcars, x = ~wt, y = ~mpg, type = 'scatter', mode = 'markers')

## lattice
# Install lattice package (run only if it’s not already installed)
if (!require("lattice")) {
  install.packages("lattice")
}

# Load lattice
library(lattice)

# Create a basic scatter plot
xyplot(mpg ~ wt, data = mtcars)

## Basic Plotting R
# Basic scatter plot
plot(mtcars$wt, mtcars$mpg,
     main = "Scatter Plot of Weight vs. Miles per Gallon",
     xlab = "Weight (1000 lbs)",
     ylab = "Miles per Gallon")
# Multiple graphs
# Define the layout with 2 rows and 2 columns
par(mfrow = c(2, 2))

# Plot 1
plot(mtcars$wt, mtcars$mpg, main = "Plot 1: Weight vs. MPG")

# Plot 2
plot(mtcars$disp, mtcars$mpg, main = "Plot 2: Displacement vs. MPG")

# Plot 3
plot(mtcars$hp, mtcars$mpg, main = "Plot 3: Horsepower vs. MPG")
# Plot 4
plot(mtcars$drat, mtcars$mpg, main = "Plot 4: Rear Axle Ratio vs. MPG")

#using gridextra package
# Install the gridExtra package (run only if it's not already installed)
if (!require("gridExtra")) {
  install.packages("gridExtra")
}

# Load necessary libraries
library(ggplot2)
library(gridExtra)

# Create individual plots
p1 <- ggplot(mtcars, aes(x = wt, y = mpg)) + 
  geom_point() + 
  ggtitle("Plot 1: Weight vs. MPG")

p2 <- ggplot(mtcars, aes(x = disp, y = mpg)) + 
  geom_point() + 
  ggtitle("Plot 2: Displacement vs. MPG")

p3 <- ggplot(mtcars, aes(x = hp, y = mpg)) + 
  geom_point() + 
  ggtitle("Plot 3: Horsepower vs. MPG")

p4 <- ggplot(mtcars, aes(x = drat, y = mpg)) + 
  geom_point() + 
  ggtitle("Plot 4: Rear Axle Ratio vs. MPG")

# Arrange the plots in a 2x2 grid
grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)

## Exporting Graphs
#base r, jpeg
# Open a JPEG device
jpeg(filename = "plot1.jpg", width = 800, height = 600)

# Create a plot
plot(mtcars$wt, mtcars$mpg, main = "Weight vs. MPG", xlab = "Weight (1000 lbs)", ylab = "Miles per Gallon")

# Close the device
dev.off()

#png
# Open a PNG device and specify file name, width, and height
png(filename = "plot1.png", width = 800, height = 600)

# Create a scatter plot
plot(
  mtcars$wt, mtcars$mpg,
  main = "Weight vs. MPG",
  xlab = "Weight (1000 lbs)",
  ylab = "Miles per Gallon"
)

# Close the PNG device to save the file
dev.off()

#pdf
# Open a PDF device
pdf(file = "plot1.pdf", width = 8, height = 6)

# Create a plot
plot(mtcars$wt, mtcars$mpg, main = "Weight vs. MPG", xlab = "Weight (1000 lbs)", ylab = "Miles per Gallon")

# Close the device
dev.off()

## exporting with ggplot2
# Create a plot
p <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  ggtitle("Weight vs. MPG")

# Save the plot as a PNG file
ggsave(filename = "plot1.png", plot = p, width = 8, height = 6, dpi = 300)

# Save the plot as jpeg
ggsave(filename = "plot1.jpg", plot = p, width = 8, height = 6, dpi = 300)

# Save the plot as pdf
ggsave(filename = "plot1.pdf", plot = p, width = 8, height = 6)
e