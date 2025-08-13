"""
Implementation of the Fibonacci sequence using memoization.
This approach uses a dictionary to store previously computed Fibonacci numbers, allowing for efficient recursive computation.
Time Complexity: O(n) – each number is computed once.
Space Complexity: O(n) – for the memo dictionary."""
def fibonacci_memo(n, memo={}):
    # Base cases
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    # Check if result is already in memo
    if n in memo:
        return memo[n]
    
    # Compute and store result in memo
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# Example usage
n = 10
print(f"Fibonacci({n}) using memoization: {fibonacci_memo(n)}")