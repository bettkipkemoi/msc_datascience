import numpy as np

# Function to compute eigenvalues and eigenvectors
def compute_eigen(matrix):
    #TODO
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

# Function for user input to get a matrix
def get_matrix_input():
    rows = int(input("Enter the number of rows in the matrix: "))
    cols = int(input("Enter the number of columns in the matrix: "))
    print("Enter the matrix row by row:")
    matrix = []
    for i in range(rows):
        row = list(map(float, input(f"Enter row {i+1}: ").split()))
        matrix.append(row)
    return np.array(matrix)

# Function to display eigenvalues and eigenvectors
def display_eigenvalues_eigenvectors(eigenvalues, eigenvectors):
    print("\nEigenvalues:")
    #TODO
    print(eigenvalues)
    print("\nEigenvectors:")
    #TODO
    print(eigenvectors)

# Function to interpret eigenvalues and eigenvectors
def interpret_eigenvalues_eigenvectors(matrix, eigenvalues, eigenvectors):
    print("\nInterpretation of Eigenvalues and Eigenvectors:")
    
    if matrix.shape[0] == matrix.shape[1]:  # Only for square matrices
        for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
            print(f"\nFor Eigenvalue {eigenvalue} and Eigenvector {eigenvector}:")
            
            # Physical and geometrical interpretation
            if eigenvalue > 0:
                #TODO
                print("The eigenvalue is positive, indicating a stretch or compression in the direction of the eigenvector.")
            elif eigenvalue < 0:
                #TODO
                print("The eigenvalue is negative, indicating a reflection or rotation in the direction of the eigenvector.")
            else:
                #TODO
                print("The eigenvalue is zero, indicating a reflection or rotation in the direction of the eigenvector.")
            
            

# Main function to run the program
def main():
    print("Eigenvalue and Eigenvector Calculator")
    matrix = get_matrix_input()
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = compute_eigen(matrix)
    
    # Display eigenvalues and eigenvectors
    display_eigenvalues_eigenvectors(eigenvalues, eigenvectors)
    
    # Interpret the eigenvalues and eigenvectors
    interpret_eigenvalues_eigenvectors(matrix, eigenvalues, eigenvectors)

if __name__ == "__main__":
    main()