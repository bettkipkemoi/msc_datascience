def counting_sort(arr, exp):
    """
    Helper function for Radix Sort: Performs counting sort on array based on digit at position exp.
    arr: Input array of non-negative integers
    exp: Exponent (10^exp) to process the current digit
    """
    n = len(arr)
    output = [0] * n  # Output array
    count = [0] * 10  # Count array for digits 0-9

    # Count occurrences of each digit
    for i in range(n):
        digit = (arr[i] // exp) % 10
        count[digit] += 1

    # Cumulative count
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build output array by placing elements in correct positions
    i = n - 1
    while i >= 0:
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1
        i -= 1

    # Copy output back to input array
    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr):
    """
    Radix Sort implementation for non-negative integers.
    arr: Input array to be sorted
    Returns: Sorted array (modifies input in-place)
    """
    if not arr:
        return arr

    # Find the maximum number to determine number of digits
    max_num = max(arr)
    exp = 1  # Start with the least significant digit (10^0)

    # Process each digit
    while max_num // exp > 0:
        counting_sort(arr, exp)
        exp *= 10

    return arr

def main():
    # Test input
    arr = [564, 213, 987, 432, 123, 765, 321, 654, 876]
    print(f"Original array: {arr}")
    
    # Sort the array
    sorted_arr = radix_sort(arr)
    print(f"Sorted array: {sorted_arr}")

if __name__ == "__main__":
    main()