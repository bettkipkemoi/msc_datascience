#if the 6th term is 13, find 23rd term
fibonacci_iterative <- function(n) {
  fib <- numeric(n)
  fib[1] <- 1  # Assume standard start, adjust via scaling
  fib[2] <- 1
  for (i in 3:n) {
    fib[i] <- fib[i - 1] + fib[i - 2]
  }
  return(fib)
}
k <- 13 / fibonacci_iterative(6)[6]  # k = 13/8 = 1.625
round(fibonacci_iterative(23)[23] * k)  # Scale and round
