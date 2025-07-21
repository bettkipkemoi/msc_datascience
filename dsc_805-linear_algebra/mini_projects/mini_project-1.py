import numpy as np

# Vector Operations
def vector_addition(v1, v2):
    #TODO
    #Vector addition
    return v1 + v2

def scalar_multiplication(v, scalar):
    #TODO
    #Vector multiplication with a scalar
    return v * scalar

def dot_product(v1, v2):
   #TODO
   #Vector multiplication with a vector
   return np.dot(v1, v2)

def cross_product(v1, v2):
    #TODO
    #Cross product
    return np.cross(v1, v2)

# Matrix Operations
def matrix_multiplication(A, B):
    #TODO
    #Matrix mltiplication
    return np.dot(A, B)

def matrix_inverse(A):
   #TODO
   #Inverse of a matrix
   return np.linalg.inv(A)

# Function to display results interactively
def interactive_system():
    print("Welcome to the Linear Algebra Interactive System!")
    
    while True:
        print("\nChoose the operation:")
        print("1. Vector Addition")
        print("2. Scalar Multiplication")
        print("3. Dot Product")
        print("4. Cross Product")
        print("5. Matrix Multiplication")
        print("6. Matrix Inversion")
        print("7. Exit")
        
        choice = int(input("\nEnter the number of the operation you want to perform: "))
        
        if choice == 1:
            # Vector Addition
            v1 = np.array(list(map(int, input("Enter vector 1 (e.g., 1 2 3): ").split())))
            v2 = np.array(list(map(int, input("Enter vector 2 (e.g., 1 2 3): ").split())))
            print("Result of vector addition:", vector_addition(v1, v2))
        
        elif choice == 2:
            # Scalar Multiplication
            v = np.array(list(map(int, input("Enter vector (e.g., 1 2 3): ").split())))
            scalar = int(input("Enter scalar value: "))
            print("Result of scalar multiplication:", scalar_multiplication(v, scalar))
        
        elif choice == 3:
            # Dot Product
            v1 = np.array(list(map(int, input("Enter vector 1 (e.g., 1 2 3): ").split())))
            v2 = np.array(list(map(int, input("Enter vector 2 (e.g., 1 2 3): ").split())))
            print("Result of dot product:", dot_product(v1, v2))
        
        elif choice == 4:
            # Cross Product (only for 3D vectors)
            v1 = np.array(list(map(int, input("Enter vector 1 (e.g., 1 2 3): ").split())))
            v2 = np.array(list(map(int, input("Enter vector 2 (e.g., 1 2 3): ").split())))
            print("Result of cross product:", cross_product(v1, v2))
        
        elif choice == 5:
            # Matrix Multiplication
            rows_a = int(input("Enter number of rows in matrix A: "))
            cols_a = int(input("Enter number of columns in matrix A: "))
            matrix_a = []
            print("Enter the elements of matrix A row by row:")
            for i in range(rows_a):
                matrix_a.append(list(map(int, input().split())))
            matrix_a = np.array(matrix_a)
            
            rows_b = int(input("Enter number of rows in matrix B: "))
            cols_b = int(input("Enter number of columns in matrix B: "))
            matrix_b = []
            print("Enter the elements of matrix B row by row:")
            for i in range(rows_b):
                matrix_b.append(list(map(int, input().split())))
            matrix_b = np.array(matrix_b)
            
            if cols_a == rows_b:
                print("Result of matrix multiplication:\n", matrix_multiplication(matrix_a, matrix_b))
            else:
                print("Matrix dimensions do not match for multiplication!")
        
        elif choice == 6:
            # Matrix Inversion
            rows = int(input("Enter number of rows in the matrix: "))
            cols = int(input("Enter number of columns in the matrix: "))
            matrix = []
            print("Enter the elements of the matrix row by row:")
            for i in range(rows):
                matrix.append(list(map(int, input().split())))
            matrix = np.array(matrix)
            
            if rows == cols:
                print("Matrix Inversion Result:\n", matrix_inverse(matrix))
            else:
                print("Matrix must be square to compute its inverse.")
        
        elif choice == 7:
            # Exit
            print("Exiting the system.")
            break
        
        else:
            print("Invalid choice! Please try again.")

# Run the interactive system
if __name__ == "__main__":
    interactive_system()
