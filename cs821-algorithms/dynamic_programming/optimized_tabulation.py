"""
Space-optimized tabulation (O(1) space) version that only stores the last two numbers:
"""
def fibonacci_tab_optimized(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr

# Example usage
n = 10
print(f"Fibonacci({n}) using optimized tabulation: {fibonacci_tab_optimized(n)}")