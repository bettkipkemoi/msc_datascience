def bubble_sort(array):
  
    # Outer loop to iterate through the list n times
    for n in range(len(array) - 1, 0, -1):
        
        # Initialize swapped to track if any swaps occur
        swapped = False  

        # Inner loop to compare adjacent elements
        for i in range(n):
            if array[i] > array[i + 1]:
              
                # Swap elements if they are in the wrong order
                array[i], array[i + 1] = array[i + 1], array[i]
                
                # Mark that a swap has occurred
                swapped = True
        
        # If no swaps occurred, the list is already sorted
        if not swapped:
            break


# Sample list to be sorted
array = [200, 189,340,520,7,89,57,1000]
print(f"Unsorted list is: {array}")

bubble_sort(array)

print(f"sorted list is: {array}")