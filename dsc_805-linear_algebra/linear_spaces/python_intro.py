#In Python, you often need to make decisions using if statements. 
#Let’s look at an example:
age = 20
if age >= 18:
    print("You are an adult.")
else:
    print("You are a minor.")

#Task 1: Modify the if-else statement for voting eligibility
age = 20
if age >= 18:
    print("You are eligible to vote.")
else:
    print("You are not eligible to vote yet.")
#If you change age to 16, it will print:
age = 16
if age >= 18:
    print("You are eligible to vote.")
else:
    print("You are not eligible to vote yet.")


#Do you want to repeat actions multiple times? 
#That’s where loops come in. Here's how a for loop works:
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Initialize the counter variable
count = 1

# Set the condition for the while loop
while count <= 10:
    # Print the current value of the counter
    print(count)
    # Increment the counter for the next iteration
    count += 1

#greet function
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
greet("Bob")

#add two numbers
def sum(a,b):
    return(a+b)    
a=4
b=6
print(f"The sum is, {sum(a,b)}")

#way 2
#Here’s how you can create a function to add two numbers:
def add_numbers(a, b):
    return a + b

# Calling the function
result = add_numbers(5, 10)
print(result)

#returning values
def square(number):
    return number ** 2

result = square(4) 
print(result)

#Variables declared inside a function are local variables and are not accessible outside the function. 
#Variables outside the function are global variables and can be accessed within the function if needed.
x = 10  # Global variable

def my_function():
    print(x)  # Access global variable

my_function()  



#You can also add to a list, remove items, or access specific values:
numbers = [1, 2, 3, 4, 5]
numbers.append(6)  # Add 6 to the list
print(numbers[0])  # Access the first item in the list (index 0)

#Let’s say you want to create a list of your favorite movies and manipulate it:
movies = ["Inception", "The Matrix", "Interstellar"]
print(movies)  # Output: ['Inception', 'The Matrix', 'Interstellar']

# Add a movie to the list
movies.append("Avatar")
print(movies)  # Output: ['Inception', 'The Matrix', 'Interstellar', 'Avatar']

# Remove a movie from the list
movies.remove("The Matrix")
print(movies)  # Output: ['Inception', 'Interstellar', 'Avatar']


#Sometimes, things go wrong. Python lets you handle errors gracefully using try and except.
#Here's a basic example:
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Oops! You can't divide by zero.")
#without try except
#result = 10/0

#Python makes it easy to work with files. Here’s how to read and write text files:
with open("bett.txt", "r") as file:
    content = file.read()
    print(content)

#Writing to a file
with open("bett.txt", "w") as file:
    file.write("This is a new line of text.")