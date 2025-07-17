#ensure that the results are reproducible
set.seed(123)

#generate random data
binomial_data <- rbinom(1000, size = 10, prob = 0.5)
poisson_data <- rpois(1000, lambda = 5)

# summary statistics and visualization
summary(binomial_data)
hist(binomial_data, main = "Histogram of Binomial Data", 
     xlab = "Value", ylab = "Frequency")

#export data
write.csv(binomial_data, "binomial_data.csv")

#example 1
#A company wants to simulate the outcome of a quality control test where each 
# product has a 90% chance of passing the test (success) and a 10% chance of 
# failing (failure).

# Set probability of success
p <- 0.9

# Simulate 1000 Bernoulli trials
set.seed(123)
bernoulli_trials <- rbinom(1000, size = 1, prob = p)

# Calculate the proportion of successes
proportion_success <- mean(bernoulli_trials)
proportion_success

## This R code simulates 1000 Bernoulli trials and calculates the proportion of successes, which should be close to 0.9.


#example 2: binomial
# A marketing campaign is run where the probability of a customer making a purchase is 0.3. If 10 customers are contacted, what is the probability that exactly 4 will make a purchase?
# Set parameters
n <- 10
p <- 0.3
k <- 4

# Calculate the probability of exactly 4 successes
prob_4_successes <- dbinom(k, size = n, prob = p)
prob_4_successes

# This R code calculates the probability of exactly 4 customers making a purchase out of 10, which is approximately 0.2001.

#example 3: poisson dist
#A call center receives an average of 5 calls per hour. What is the probability that exactly 3 calls will be received in an hour?
# Set parameters
lambda <- 5
k <- 3

# Calculate the probability of exactly 3 calls
prob_3_calls <- dpois(k, lambda = lambda)
prob_3_calls

# This R code calculates the probability of receiving exactly 3 calls in an hour, which is approximately 0.1404.

#normal distribution

#Generate 1000 random samples from a normal distribution with a mean of 50 and a standard deviation of 10. Plot a histogram of the samples and calculate the sample mean and standard deviation.
# Here is an example code in R to perform this simulation:
# Set seed for reproducibility
set.seed(123)

# Parameters
mean <- 50
sd <- 10
n <- 1000

# Generate random samples
samples <- rnorm(n, mean = mean, sd = sd)

# Plot histogram
hist(samples, breaks = 30, main = "Histogram of Normal Distribution",
     xlab = "Value", ylab = "Frequency", col = "lightblue", border = "black")

# Calculate sample statistics
sample_mean <- mean(samples)
sample_sd <- sd(samples)

# Print sample statistics
cat("Sample Mean:", sample_mean, "\n")
cat("Sample Standard Deviation:", sample_sd, "\n")
