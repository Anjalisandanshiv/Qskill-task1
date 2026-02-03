"""
Matrix Operations Tool - Demonstration Script
Shows all available operations with example matrices
"""

import numpy as np
import sys

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)

def print_matrix(matrix, name="Matrix"):
    """Print a matrix in a formatted way"""
    print(f"\n{name}:")
    print("-" * 60)
    for row in matrix:
        print("  [", " ".join([f"{val:>8.3f}" for val in row]), "]")
    print(f"Shape: {matrix.shape}")
    print("-" * 60)

def print_scalar(value, name="Result"):
    """Print a scalar result"""
    print(f"\n{name}: {value:.6f}")
    print("-" * 60)

# Main demonstration
print_header("MATRIX OPERATIONS TOOL - COMPREHENSIVE DEMONSTRATION")
print("\nThis script demonstrates all available matrix operations")
print("using NumPy with example matrices.")

# Define example matrices
print_header("EXAMPLE MATRICES")

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

B = np.array([
    [9, 8, 7],
    [6, 5, 4],
    [3, 2, 1]
])

C = np.array([
    [2, -1],
    [1, 3]
])

D = np.array([
    [4, 0],
    [1, 2]
])

print_matrix(A, "Matrix A (3×3)")
print_matrix(B, "Matrix B (3×3)")
print_matrix(C, "Matrix C (2×2)")
print_matrix(D, "Matrix D (2×2)")

# 1. MATRIX ADDITION
print_header("1. MATRIX ADDITION (A + B)")
print("\nOperation: Add corresponding elements of two matrices")
print("Requirement: Matrices must have same dimensions")
result = A + B
print_matrix(A, "A")
print_matrix(B, "B")
print_matrix(result, "Result: A + B")

# 2. MATRIX SUBTRACTION
print_header("2. MATRIX SUBTRACTION (A - B)")
print("\nOperation: Subtract corresponding elements")
print("Requirement: Matrices must have same dimensions")
result = A - B
print_matrix(A, "A")
print_matrix(B, "B")
print_matrix(result, "Result: A - B")

# 3. MATRIX MULTIPLICATION
print_header("3. MATRIX MULTIPLICATION (C × D)")
print("\nOperation: Multiply matrices using dot product")
print("Requirement: Number of columns in first matrix = number of rows in second")
print(f"C shape: {C.shape}, D shape: {D.shape}")
result = np.matmul(C, D)
print_matrix(C, "C (2×2)")
print_matrix(D, "D (2×2)")
print_matrix(result, "Result: C × D (2×2)")
print("\nCalculation:")
print(f"  [2×4 + (-1)×1   2×0 + (-1)×2]   =   [{result[0,0]:.1f}  {result[0,1]:.1f}]")
print(f"  [1×4 +   3×1    1×0 +   3×2 ]       [{result[1,0]:.1f}  {result[1,1]:.1f}]")

# 4. ELEMENT-WISE MULTIPLICATION
print_header("4. ELEMENT-WISE MULTIPLICATION (A ⊙ B)")
print("\nOperation: Multiply corresponding elements (Hadamard product)")
print("Requirement: Matrices must have same dimensions")
result = A * B
print_matrix(A, "A")
print_matrix(B, "B")
print_matrix(result, "Result: A ⊙ B")

# 5. SCALAR MULTIPLICATION
print_header("5. SCALAR MULTIPLICATION (3 × C)")
print("\nOperation: Multiply every element by a scalar")
scalar = 3
result = scalar * C
print(f"Scalar: {scalar}")
print_matrix(C, "C")
print_matrix(result, f"Result: {scalar} × C")

# 6. TRANSPOSE
print_header("6. TRANSPOSE (Aᵀ)")
print("\nOperation: Swap rows and columns")
print("Property: (Aᵀ)ᵀ = A")
result = A.T
print_matrix(A, "A (3×3)")
print_matrix(result, "Result: Aᵀ (3×3)")

# 7. DETERMINANT
print_header("7. DETERMINANT (|C|)")
print("\nOperation: Scalar value representing matrix properties")
print("Requirement: Square matrix only")
print("Property: |AB| = |A| × |B|")
det_C = np.linalg.det(C)
det_D = np.linalg.det(D)
print_matrix(C, "C")
print_scalar(det_C, "Determinant of C")
print_matrix(D, "D")
print_scalar(det_D, "Determinant of D")

# 8. INVERSE
print_header("8. MATRIX INVERSE (C⁻¹)")
print("\nOperation: Find matrix such that C × C⁻¹ = I")
print("Requirement: Square matrix with non-zero determinant")
print(f"Determinant of C: {det_C:.6f} (non-zero, inverse exists)")
inv_C = np.linalg.inv(C)
print_matrix(C, "C")
print_matrix(inv_C, "C⁻¹")

# Verify: C × C⁻¹ = I
verification = np.matmul(C, inv_C)
print("\nVerification: C × C⁻¹ =")
print_matrix(verification, "Result (should be Identity)")

# 9. RANK
print_header("9. MATRIX RANK")
print("\nOperation: Number of linearly independent rows/columns")
print("Property: rank(A) ≤ min(rows, cols)")
rank_A = np.linalg.matrix_rank(A)
rank_C = np.linalg.matrix_rank(C)
print_matrix(A, "A (3×3)")
print(f"\nRank of A: {rank_A}")
print_matrix(C, "C (2×2)")
print(f"\nRank of C: {rank_C}")

# 10. TRACE
print_header("10. TRACE")
print("\nOperation: Sum of diagonal elements")
print("Requirement: Square matrix only")
print("Property: Tr(A + B) = Tr(A) + Tr(B)")
trace_A = np.trace(A)
trace_C = np.trace(C)
print_matrix(A, "A")
print(f"\nTrace of A: {trace_A:.3f} (sum: 1 + 5 + 9)")
print_matrix(C, "C")
print(f"\nTrace of C: {trace_C:.3f} (sum: 2 + 3)")

# 11. EIGENVALUES AND EIGENVECTORS
print_header("11. EIGENVALUES & EIGENVECTORS")
print("\nOperation: Find λ and v such that Av = λv")
print("Requirement: Square matrix")
eigenvalues, eigenvectors = np.linalg.eig(C)
print_matrix(C, "C")
print("\nEigenvalues (λ):")
print("-" * 60)
for i, val in enumerate(eigenvalues):
    if np.isreal(val):
        print(f"  λ{i+1} = {val.real:.6f}")
    else:
        print(f"  λ{i+1} = {val.real:.6f} + {val.imag:.6f}i")

print("\nEigenvectors:")
print_matrix(eigenvectors, "Eigenvector Matrix")
print("\nNote: Each column is an eigenvector corresponding to an eigenvalue")

# 12. MATRIX POWER
print_header("12. MATRIX POWER (C³)")
print("\nOperation: Multiply matrix by itself n times")
print("Requirement: Square matrix")
power = 3
result = np.linalg.matrix_power(C, power)
print_matrix(C, "C")
print_matrix(result, f"C^{power}")

# Show C², then C³ = C² × C
C_squared = np.linalg.matrix_power(C, 2)
print("\nStep-by-step:")
print_matrix(C_squared, "C² = C × C")
print_matrix(result, "C³ = C² × C")

# ADDITIONAL OPERATIONS
print_header("ADDITIONAL USEFUL OPERATIONS")

# 13. Identity Matrix
print("\n13. IDENTITY MATRIX:")
I = np.eye(4)
print_matrix(I, "4×4 Identity Matrix")

# 14. Zero Matrix
print("\n14. ZERO MATRIX:")
Z = np.zeros((3, 4))
print_matrix(Z, "3×4 Zero Matrix")

# 15. Random Matrix
print("\n15. RANDOM MATRIX:")
np.random.seed(42)
R = np.random.randint(1, 10, (3, 3))
print_matrix(R, "3×3 Random Matrix (integers 1-9)")

# 16. Matrix Norms
print_header("16. MATRIX NORMS")
print("\nDifferent ways to measure matrix 'size':")
print_matrix(C, "C")
print(f"\nFrobenius Norm (||C||_F): {np.linalg.norm(C, 'fro'):.6f}")
print(f"Infinity Norm (||C||_∞):   {np.linalg.norm(C, np.inf):.6f}")
print(f"1-Norm (||C||_1):          {np.linalg.norm(C, 1):.6f}")

# 17. Condition Number
print_header("17. CONDITION NUMBER")
print("\nMeasures sensitivity to numerical errors:")
cond_C = np.linalg.cond(C)
print_matrix(C, "C")
print(f"\nCondition Number: {cond_C:.6f}")
print("(Lower is better; values > 1e10 indicate ill-conditioned matrix)")

# 18. Reshape and Flatten
print_header("18. RESHAPE & FLATTEN")
original = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print_matrix(original, "Original (2×4)")
reshaped = original.reshape(4, 2)
print_matrix(reshaped, "Reshaped (4×2)")
flattened = original.flatten()
print(f"\nFlattened: {flattened}")

# 19. Concatenation
print_header("19. MATRIX CONCATENATION")
M1 = np.array([[1, 2], [3, 4]])
M2 = np.array([[5, 6], [7, 8]])
print_matrix(M1, "M1")
print_matrix(M2, "M2")
horizontal = np.hstack([M1, M2])
print_matrix(horizontal, "Horizontal Concatenation [M1 | M2]")
vertical = np.vstack([M1, M2])
print_matrix(vertical, "Vertical Concatenation [M1; M2]")

# 20. Statistical Operations
print_header("20. STATISTICAL OPERATIONS")
print_matrix(A, "A")
print(f"\nSum of all elements:     {np.sum(A):.3f}")
print(f"Mean of all elements:    {np.mean(A):.3f}")
print(f"Standard deviation:      {np.std(A):.3f}")
print(f"Maximum value:           {np.max(A):.3f}")
print(f"Minimum value:           {np.min(A):.3f}")
print(f"\nRow sums:    {np.sum(A, axis=1)}")
print(f"Column sums: {np.sum(A, axis=0)}")

# SUMMARY
print_header("OPERATION SUMMARY")
print("""
✓ Basic Operations:
  • Addition, Subtraction, Multiplication
  • Element-wise operations
  • Scalar multiplication

✓ Matrix Properties:
  • Transpose, Determinant, Inverse
  • Rank, Trace
  • Eigenvalues & Eigenvectors

✓ Advanced Operations:
  • Matrix Power
  • Norms & Condition Number
  • Reshape & Concatenation
  • Statistical operations

✓ Matrix Creation:
  • Identity, Zero, Random matrices
  • Custom dimensions
  
All operations demonstrated with NumPy library!
""")

print("\n" + "=" * 80)
print("DEMONSTRATION COMPLETE".center(80))
print("=" * 80 + "\n")
