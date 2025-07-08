def calculate_average():
    sum_values = 0
    count = 0
    
    num_values = int(input("Enter the number of values: "))    
    for i in range(num_values):
        value = float(input(f"Enter value {i+1}: "))
        count += 1
        sum_values += value
        
    if count > 0:
        average = sum_values / count
        print(f"The average is: {average:.2f}")
    else:
        print("No values entered.")

# Run the program
if __name__ == "__main__":
    calculate_average()