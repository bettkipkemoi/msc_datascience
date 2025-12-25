def check_even(num):
    if num == 0:
        result = "Zero"
    elif num % 2 == 0:
        result = "Even number"
    else:
        result = "Odd number"
    return result

print(check_even(4))  # Output: Even number
print(check_even(7))  # Output: Odd number
print(check_even(0))  # Output: Zero
# This function checks if a number is even, odd, or zero and returns a corresponding message
