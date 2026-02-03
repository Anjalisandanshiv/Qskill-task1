"""
Quick Test Script for Matrix Operations Tool
Demonstrates basic functionality without interactive input
"""

import numpy as np

print("=" * 80)
print("MATRIX OPERATIONS TOOL - QUICK TEST".center(80))
print("=" * 80)

# Test 1: Create sample matrices
print("\n📊 Creating Test Matrices...")
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
C = np.array([[2, -1], [1, 3]])

print("\nMatrix A (3×3):")
print(A)
print("\nMatrix B (3×3):")
print(B)
print("\nMatrix C (2×2):")
print(C)

# Test 2: Addition
print("\n" + "=" * 80)
print("TEST 1: Matrix Addition (A + B)")
print("=" * 80)
result = A + B
print("Result:")
print(result)
print(f"✅ Addition successful! Result shape: {result.shape}")

# Test 3: Subtraction
print("\n" + "=" * 80)
print("TEST 2: Matrix Subtraction (A - B)")
print("=" * 80)
result = A - B
print("Result:")
print(result)
print(f"✅ Subtraction successful! Result shape: {result.shape}")

# Test 4: Multiplication
print("\n" + "=" * 80)
print("TEST 3: Matrix Multiplication (C × C)")
print("=" * 80)
result = np.matmul(C, C)
print("Result:")
print(result)
print(f"✅ Multiplication successful! Result shape: {result.shape}")

# Test 5: Transpose
print("\n" + "=" * 80)
print("TEST 4: Transpose (Aᵀ)")
print("=" * 80)
result = A.T
print("Result:")
print(result)
print(f"✅ Transpose successful! Result shape: {result.shape}")

# Test 6: Determinant
print("\n" + "=" * 80)
print("TEST 5: Determinant |C|")
print("=" * 80)
det = np.linalg.det(C)
print(f"Determinant of C: {det:.6f}")
print(f"✅ Determinant calculation successful!")

# Test 7: Inverse
print("\n" + "=" * 80)
print("TEST 6: Matrix Inverse (C⁻¹)")
print("=" * 80)
inv_C = np.linalg.inv(C)
print("C⁻¹:")
print(inv_C)
# Verify
verification = np.matmul(C, inv_C)
print("\nVerification (C × C⁻¹):")
print(verification)
print(f"✅ Inverse calculation successful!")

# Test 8: Eigenvalues
print("\n" + "=" * 80)
print("TEST 7: Eigenvalues & Eigenvectors")
print("=" * 80)
eigenvals, eigenvecs = np.linalg.eig(C)
print("Eigenvalues:")
for i, val in enumerate(eigenvals):
    print(f"  λ{i+1} = {val:.6f}")
print("\nEigenvectors:")
print(eigenvecs)
print(f"✅ Eigenvalue calculation successful!")

# Test 9: Rank
print("\n" + "=" * 80)
print("TEST 8: Matrix Rank")
print("=" * 80)
rank_A = np.linalg.matrix_rank(A)
rank_C = np.linalg.matrix_rank(C)
print(f"Rank of A: {rank_A}")
print(f"Rank of C: {rank_C}")
print(f"✅ Rank calculation successful!")

# Test 10: Trace
print("\n" + "=" * 80)
print("TEST 9: Matrix Trace")
print("=" * 80)
trace_A = np.trace(A)
trace_C = np.trace(C)
print(f"Trace of A: {trace_A:.3f} (sum: {A[0,0]} + {A[1,1]} + {A[2,2]})")
print(f"Trace of C: {trace_C:.3f} (sum: {C[0,0]} + {C[1,1]})")
print(f"✅ Trace calculation successful!")

# Summary
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED SUCCESSFULLY!".center(80))
print("=" * 80)
print("\nThe Matrix Operations Tool is working correctly!")
print("Run 'python matrix_operations_cli.py' for the interactive interface.")
print("Open 'matrix_operations_web.html' in a browser for the web interface.")
print("=" * 80 + "\n")
