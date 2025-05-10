"""
Introduction to NumPy

This script provides a comprehensive introduction to NumPy, covering:
- Array creation and manipulation
- Array operations and broadcasting
- Array indexing and slicing
- Mathematical operations
- Statistical functions
- Linear algebra operations
"""

import numpy as np
import matplotlib.pyplot as plt

def array_creation():
    """Demonstrate different ways to create NumPy arrays"""
    print("\n=== Array Creation ===")
    
    # Create arrays from lists
    arr1 = np.array([1, 2, 3, 4, 5])
    print("1D array from list:", arr1)
    
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    print("\n2D array from nested lists:\n", arr2)
    
    # Special arrays
    zeros = np.zeros((3, 3))
    print("\nArray of zeros:\n", zeros)
    
    ones = np.ones((2, 4))
    print("\nArray of ones:\n", ones)
    
    identity = np.eye(3)
    print("\nIdentity matrix:\n", identity)
    
    # Arrays with specific ranges
    range_arr = np.arange(0, 10, 2)
    print("\nArray with range:\n", range_arr)
    
    linspace = np.linspace(0, 1, 5)
    print("\nLinearly spaced array:\n", linspace)
    
    # Random arrays
    random = np.random.rand(3, 3)
    print("\nRandom array:\n", random)
    
    return arr1, arr2, zeros, ones, identity, range_arr, linspace, random

def array_attributes():
    """Demonstrate NumPy array attributes"""
    print("\n=== Array Attributes ===")
    
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    
    print("Array:\n", arr)
    print("Shape:", arr.shape)
    print("Number of dimensions:", arr.ndim)
    print("Size (total elements):", arr.size)
    print("Data type:", arr.dtype)
    print("Item size (bytes):", arr.itemsize)
    print("Total bytes:", arr.nbytes)
    
    return arr

def array_operations():
    """Demonstrate basic array operations"""
    print("\n=== Array Operations ===")
    
    # Element-wise operations
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    print("Addition:", a + b)
    print("Subtraction:", a - b)
    print("Multiplication:", a * b)
    print("Division:", a / b)
    print("Power:", a ** 2)
    
    # Broadcasting
    c = np.array([[1, 2, 3], [4, 5, 6]])
    print("\nBroadcasting addition:\n", c + 10)
    
    # Matrix multiplication
    d = np.array([[1, 2], [3, 4]])
    e = np.array([[5, 6], [7, 8]])
    print("\nMatrix multiplication:\n", np.dot(d, e))
    
    return a, b, c, d, e

def array_indexing():
    """Demonstrate array indexing and slicing"""
    print("\n=== Array Indexing and Slicing ===")
    
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print("Original array:\n", arr)
    
    # Basic indexing
    print("\nElement at (0,0):", arr[0, 0])
    print("First row:", arr[0])
    print("First column:", arr[:, 0])
    
    # Slicing
    print("\nFirst two rows:\n", arr[:2])
    print("Last two columns:\n", arr[:, -2:])
    print("Middle subarray:\n", arr[1:3, 1:3])
    
    # Boolean indexing
    mask = arr > 5
    print("\nElements greater than 5:\n", arr[mask])
    
    # Fancy indexing
    indices = [0, 2]
    print("\nSelected rows:\n", arr[indices])
    
    return arr

def mathematical_operations():
    """Demonstrate mathematical operations"""
    print("\n=== Mathematical Operations ===")
    
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print("Original array:\n", arr)
    
    # Basic math
    print("\nSum:", np.sum(arr))
    print("Mean:", np.mean(arr))
    print("Standard deviation:", np.std(arr))
    print("Minimum:", np.min(arr))
    print("Maximum:", np.max(arr))
    
    # Axis operations
    print("\nSum along rows:", np.sum(arr, axis=0))
    print("Sum along columns:", np.sum(arr, axis=1))
    
    # Trigonometric functions
    angles = np.array([0, np.pi/2, np.pi])
    print("\nSine:", np.sin(angles))
    print("Cosine:", np.cos(angles))
    
    return arr

def statistical_functions():
    """Demonstrate statistical functions"""
    print("\n=== Statistical Functions ===")
    
    # Generate random data
    data = np.random.normal(0, 1, 1000)
    
    # Basic statistics
    print("Mean:", np.mean(data))
    print("Median:", np.median(data))
    print("Standard deviation:", np.std(data))
    print("Variance:", np.var(data))
    print("25th percentile:", np.percentile(data, 25))
    print("75th percentile:", np.percentile(data, 75))
    
    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.7)
    plt.title('Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('normal_distribution.png')
    plt.close()
    
    return data

def linear_algebra():
    """Demonstrate linear algebra operations"""
    print("\n=== Linear Algebra ===")
    
    # Matrix operations
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print("Matrix A:\n", A)
    print("\nMatrix B:\n", B)
    
    # Basic operations
    print("\nMatrix addition:\n", A + B)
    print("\nMatrix multiplication:\n", np.dot(A, B))
    
    # Advanced operations
    print("\nDeterminant of A:", np.linalg.det(A))
    print("\nInverse of A:\n", np.linalg.inv(A))
    print("\nEigenvalues of A:", np.linalg.eigvals(A))
    
    # Solve linear equations
    b = np.array([1, 2])
    x = np.linalg.solve(A, b)
    print("\nSolution to Ax = b:", x)
    
    return A, B

def broadcasting_example():
    """Demonstrate broadcasting with a practical example"""
    print("\n=== Broadcasting Example ===")
    
    # Create a 2D grid
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distance from origin
    Z = np.sqrt(X**2 + Y**2)
    
    # Plot the result
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar()
    plt.title('Distance from Origin')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('broadcasting_example.png')
    plt.close()
    
    return X, Y, Z

def main():
    """Main function to demonstrate NumPy features"""
    print("=== Introduction to NumPy ===")
    
    # Array creation
    print("\n=== Array Creation Examples ===")
    array_creation()
    
    # Array attributes
    print("\n=== Array Attributes Examples ===")
    array_attributes()
    
    # Array operations
    print("\n=== Array Operations Examples ===")
    array_operations()
    
    # Array indexing
    print("\n=== Array Indexing Examples ===")
    array_indexing()
    
    # Mathematical operations
    print("\n=== Mathematical Operations Examples ===")
    mathematical_operations()
    
    # Statistical functions
    print("\n=== Statistical Functions Examples ===")
    statistical_functions()
    
    # Linear algebra
    print("\n=== Linear Algebra Examples ===")
    linear_algebra()
    
    # Broadcasting example
    print("\n=== Broadcasting Example ===")
    broadcasting_example()
    
    print("\nAll NumPy examples completed successfully!")
    print("Visualizations have been saved as PNG files.")

if __name__ == "__main__":
    main() 