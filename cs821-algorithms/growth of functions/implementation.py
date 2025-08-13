import random
import time

# Linear Search
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Bubble Sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Function to generate random dataset
def generate_dataset(size):
    return [random.randint(1, 10000) for _ in range(size)]

# Function to measure execution time
def measure_time(algorithm, dataset, target=None):
    start_time = time.time()
    if algorithm == linear_search:
        result = linear_search(dataset, target)
    else:  # bubble_sort
        result = bubble_sort(dataset.copy())  # Copy to avoid modifying original
    end_time = time.time()
    return (end_time - start_time) * 1000  # Convert to milliseconds

# Test setup
input_sizes = [10, 100, 1000, 10000]
num_runs = 10  # Number of runs to average
linear_times = []
bubble_times = []

# Run tests
for size in input_sizes:
    linear_avg_time = 0
    bubble_avg_time = 0
    dataset = generate_dataset(size)
    target = random.choice(dataset)  # Random target for linear search
    
    for _ in range(num_runs):
        # Measure Linear Search time
        linear_avg_time += measure_time(linear_search, dataset, target)
        # Measure Bubble Sort time
        bubble_avg_time += measure_time(bubble_sort, dataset)
    
    # Average the times
    linear_avg_time /= num_runs
    bubble_avg_time /= num_runs
    linear_times.append(linear_avg_time)
    bubble_times.append(bubble_avg_time)