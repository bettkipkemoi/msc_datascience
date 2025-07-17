#binomial

#Suppose a factory produces light bulbs, and the probability that a randomly selected bulb is defective is 0.05. If a quality control inspector randomly selects 20 bulbs from the production line, what is the probability that exactly 2 of the selected bulbs are defective?

# Parameters
n <- 20   # number of trials
p <- 0.05  # probability of success

# Probability of getting exactly 5 successes
k <- 2
prob <- dbinom(k, n, p)
print(prob)
# The Probability of 2 defective bulbs is 0.1886768

# if interest is P(X<=2), we find 3 probs, k=0,1,2
#Based on the information above
prob <- dbinom(0, 20, 0.05) + dbinom(1, 20, 0.05) + dbinom(2, 20, 0.05)

# The probability of 2 or fewer defective bulbs is 0.9245163

#simulations to generate binomial random variables using the rbinom function
# Parameters
n <- 10   # number of trials
p <- 0.5  # probability of success
size <- 1000  # number of simulations

# Simulating binomial random variables
sim_data <- rbinom(size, n, p)

# Displaying the first 10 simulated values
print(head(sim_data, 10))

#calculate the probability of getting exactly 3 successes in 5 trials with a success probability of 0.7, and simulate 10,000 observations
# Parameters for probability calculation
n <- 5
p <- 0.7
k <- 3

# Probability calculation
prob <- dbinom(k, n, p)
print(paste("Probability of exactly 3 successes:", prob))

# Parameters for simulation
size <- 10000

# Simulating binomial random variables
sim_data <- rbinom(size, n, p)


## Poisson distribution
#Calculate the probability of observing exactly 3 events in a Poisson distribution with an average rate of 2 events per interval.

# Parameters
lambda <- 2
k <- 3

# Calculate probability
probability <- dpois(k, lambda)
print(probability)

## simulating poisson distribution rpois function
# Parameters
lambda <- 2
n <- 10

# Generate random variables
random_variables <- rpois(n, lambda)
print(random_variables)


## calculating probabilities
#dnorm(x, mean, sd): Computes the density (height of the probability density function) at x for a normal distribution with a specified mean and sd (standard deviation).
#pnorm(q, mean, sd): Computes the cumulative probability up to q for a normal distribution with a specified mean and sd.
#qnorm(p, mean, sd): Computes the quantile (the inverse of the cumulative distribution function) at p for a normal distribution with a specified mean and sd.
#rnorm(n, mean, sd): Generates n random samples from a normal distribution with a specified mean and sd.

# Set parameters
mean <- 0
sd <- 1

# Calculate probability density at x = 1
density <- dnorm(1, mean, sd)

# Calculate cumulative probability up to x = 1
cumulative_prob <- pnorm(1, mean, sd)

# Calculate the 95th percentile
percentile_95 <- qnorm(0.95, mean, sd)

# Generate 10 random samples
samples <- rnorm(10, mean, sd)

# Print results
cat("Probability density at x = 1:", density, "\n")
cat("Cumulative probability up to x = 1:", cumulative_prob, "\n")
cat("95th percentile:", percentile_95, "\n")
cat("Random samples:", samples, "\n")


## simulation using normal rnorm
# Set parameters
mean <- 0
sd <- 1
n <- 1000

# Generate random samples
samples <- rnorm(n, mean, sd)

# Plot histogram
hist(samples, breaks = 30, main = "Histogram of Random Samples", 
     xlab = "Value", ylab = "Frequency", col = "lightblue")

# probability of selecting an employee >75000
mean_salary <- 60000
sd_salary <- 15000
salary <- 75000

# Probability of earning more than 75,000
probability <- 1 - pnorm(salary, mean_salary, sd_salary)
print(probability)

# quantile function of normal distribution
percentile <- 0.90

# Salary at the 90th percentile
salary_90th <- qnorm(percentile, mean_salary, sd_salary)
print(salary_90th)

#simulate 1000 employees and plot histogram
n <- 1000

# Simulate salaries
simulated_salaries <- rnorm(n, mean_salary, sd_salary)

# Plot histogram
hist(simulated_salaries, breaks = 30, main = "Histogram of Simulated Salaries", 
     xlab = "Salary", col = "lightblue")


# calculate t-distribution
# Calculate the cumulative probability P(T <= t) for t = 1.5 with df = 10
prob <- pt(1.5, df = 10)
prob

# Generate 1000 random samples from a Student's t-distribution with df = 10
set.seed(123) # Setting seed for reproducibility
samples <- rt(1000, df = 10)
hist(samples, main = "Histogram of Student's t-distribution samples", 
     xlab = "Value", ylab = "Frequency")

#calculate uniform distribution U(a,b)
# Parameters
a <- 0
b <- 1

# Probability calculation
p <- punif(0.5, min = a, max = b)
p

#simulate uniform dist
# Number of simulations
n <- 1000

# Simulating data
sim_data <- runif(n, min = 0, max = 1)

# Displaying first few values
head(sim_data)


# exponential distribution

#suppose lambda = 0.5, we want to find P(X<=2)
lambda <- 0.5
x <- 2
prob <- pexp(x, rate = lambda)
prob

#Simulate exp dist
lambda <- 0.5
n <- 1000
sim_data <- rexp(n, rate = lambda)
hist(sim_data, breaks = 50, main = "Histogram of Simulated Exponential Data", xlab = "Value")




## Monte-Carlo Simulation
#Simulate rolling a fair six-sided die 10,000 times and estimate the probability of rolling a 4.

  #Generate 10,000 random numbers between 1 and 6.
  #Count the number of times the number 4 appears.  
  #Estimate the probability by dividing the count of 4s by the total number of rolls.
set.seed(123)
rolls <- sample(1:6, 10000, replace = TRUE)
prob_4 <- sum(rolls == 4) / 10000
prob_4


#monte carlo
set.seed(123)  # For reproducibility
nrep <- 2000
x <- runif(nrep, 0, 1)
integral_estimate <- mean(x^2)

set.seed(123)  # For reproducibility
nrep <- 4000
x <- runif(nrep, 1, pi)
integral_estimate <- (pi - 1) * mean(exp(x))


# Parameters
lambda <- 21  # Average number of calls per hour
k <- 39       # Number of calls to find probability for

# Calculate P(X = 39)
prob <- dpois(k, lambda)

# Output the result
print(prob)

n <- 1600
k <- 30
p <- 0.02
prob <- dbinom(k, n, p)
print(prob)
