"""
Matrix Operations Tool - Command Line Interface
A comprehensive tool for performing various matrix operations using NumPy
"""

import numpy as np
import sys
from typing import Optional, Tuple

class MatrixOperationsTool:
    """A comprehensive tool for matrix operations"""
    
    def __init__(self):
        self.matrix_storage = {}
        self.operation_history = []
    
    def display_header(self):
        """Display the application header"""
        print("\n" + "=" * 70)
        print("MATRIX OPERATIONS TOOL".center(70))
        print("=" * 70)
        print("Powered by NumPy | Interactive Matrix Calculator".center(70))
        print("=" * 70 + "\n")
    
    def display_matrix(self, matrix: np.ndarray, title: str = "Matrix"):
        """Display a matrix in a formatted way"""
        print(f"\n{'─' * 70}")
        print(f"📊 {title}")
        print(f"{'─' * 70}")
        print(f"Shape: {matrix.shape}")
        print(f"Data type: {matrix.dtype}")
        print(f"\nMatrix values:")
        
        # Format the matrix display
        if matrix.size <= 100:  # Only display if not too large
            for i, row in enumerate(matrix):
                row_str = " | ".join([f"{val:>10.4f}" if isinstance(val, (float, np.floating)) 
                                      else f"{val:>10}" for val in (row if matrix.ndim > 1 else [row])])
                print(f"  [{row_str} ]")
        else:
            print(f"  (Matrix too large to display - {matrix.shape})")
        print(f"{'─' * 70}")
    
    def input_matrix(self, name: str = "Matrix") -> Optional[np.ndarray]:
        """Interactive matrix input with multiple methods"""
        print(f"\n{'═' * 70}")
        print(f"📝 Enter {name}")
        print(f"{'═' * 70}")
        print("\nChoose input method:")
        print("  1. Enter values manually (row by row)")
        print("  2. Enter as space-separated rows (e.g., '1 2 3; 4 5 6')")
        print("  3. Generate random matrix")
        print("  4. Generate identity matrix")
        print("  5. Generate zero matrix")
        print("  6. Load from storage")
        print("  0. Cancel")
        
        choice = input("\nSelect option (0-6): ").strip()
        
        try:
            if choice == '1':
                return self._input_manual()
            elif choice == '2':
                return self._input_string()
            elif choice == '3':
                return self._generate_random()
            elif choice == '4':
                return self._generate_identity()
            elif choice == '5':
                return self._generate_zero()
            elif choice == '6':
                return self._load_from_storage()
            elif choice == '0':
                return None
            else:
                print("❌ Invalid option!")
                return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def _input_manual(self) -> np.ndarray:
        """Manual matrix entry row by row"""
        rows = int(input("\nEnter number of rows: "))
        cols = int(input("Enter number of columns: "))
        
        matrix_data = []
        print(f"\nEnter {rows} rows (space-separated values):")
        
        for i in range(rows):
            while True:
                row_input = input(f"  Row {i+1}: ").strip().split()
                if len(row_input) == cols:
                    matrix_data.append([float(x) for x in row_input])
                    break
                else:
                    print(f"    ⚠️  Expected {cols} values, got {len(row_input)}. Try again.")
        
        return np.array(matrix_data)
    
    def _input_string(self) -> np.ndarray:
        """Input matrix as string (MATLAB style)"""
        print("\nEnter matrix (e.g., '1 2 3; 4 5 6; 7 8 9'):")
        matrix_str = input("  → ").strip()
        
        rows = matrix_str.split(';')
        matrix_data = []
        
        for row in rows:
            row_values = [float(x) for x in row.strip().split()]
            matrix_data.append(row_values)
        
        return np.array(matrix_data)
    
    def _generate_random(self) -> np.ndarray:
        """Generate random matrix"""
        rows = int(input("\nEnter number of rows: "))
        cols = int(input("Enter number of columns: "))
        min_val = float(input("Enter minimum value (default 0): ") or "0")
        max_val = float(input("Enter maximum value (default 10): ") or "10")
        
        matrix = np.random.uniform(min_val, max_val, (rows, cols))
        print(f"\n✅ Generated {rows}×{cols} random matrix")
        return matrix
    
    def _generate_identity(self) -> np.ndarray:
        """Generate identity matrix"""
        size = int(input("\nEnter matrix size: "))
        matrix = np.eye(size)
        print(f"\n✅ Generated {size}×{size} identity matrix")
        return matrix
    
    def _generate_zero(self) -> np.ndarray:
        """Generate zero matrix"""
        rows = int(input("\nEnter number of rows: "))
        cols = int(input("Enter number of columns: "))
        matrix = np.zeros((rows, cols))
        print(f"\n✅ Generated {rows}×{cols} zero matrix")
        return matrix
    
    def _load_from_storage(self) -> Optional[np.ndarray]:
        """Load matrix from storage"""
        if not self.matrix_storage:
            print("\n❌ No matrices in storage!")
            return None
        
        print("\n📦 Stored Matrices:")
        for name in self.matrix_storage:
            shape = self.matrix_storage[name].shape
            print(f"  • {name}: {shape}")
        
        name = input("\nEnter matrix name to load: ").strip()
        if name in self.matrix_storage:
            print(f"✅ Loaded matrix '{name}'")
            return self.matrix_storage[name]
        else:
            print("❌ Matrix not found!")
            return None
    
    def store_matrix(self, matrix: np.ndarray, name: str):
        """Store matrix with a name"""
        self.matrix_storage[name] = matrix.copy()
        print(f"✅ Matrix stored as '{name}'")
    
    def add_matrices(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Add two matrices"""
        if A.shape != B.shape:
            raise ValueError(f"Matrix dimensions don't match: {A.shape} vs {B.shape}")
        result = A + B
        self.operation_history.append("Addition")
        return result
    
    def subtract_matrices(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Subtract two matrices"""
        if A.shape != B.shape:
            raise ValueError(f"Matrix dimensions don't match: {A.shape} vs {B.shape}")
        result = A - B
        self.operation_history.append("Subtraction")
        return result
    
    def multiply_matrices(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Multiply two matrices"""
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Cannot multiply: {A.shape} × {B.shape} (inner dimensions don't match)")
        result = np.matmul(A, B)
        self.operation_history.append("Multiplication")
        return result
    
    def element_wise_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Element-wise multiplication (Hadamard product)"""
        if A.shape != B.shape:
            raise ValueError(f"Matrix dimensions don't match: {A.shape} vs {B.shape}")
        result = A * B
        self.operation_history.append("Element-wise multiplication")
        return result
    
    def scalar_multiply(self, A: np.ndarray, scalar: float) -> np.ndarray:
        """Multiply matrix by scalar"""
        result = A * scalar
        self.operation_history.append(f"Scalar multiplication (×{scalar})")
        return result
    
    def transpose(self, A: np.ndarray) -> np.ndarray:
        """Transpose a matrix"""
        result = A.T
        self.operation_history.append("Transpose")
        return result
    
    def determinant(self, A: np.ndarray) -> float:
        """Calculate determinant"""
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Determinant requires square matrix, got {A.shape}")
        det = np.linalg.det(A)
        self.operation_history.append("Determinant")
        return det
    
    def inverse(self, A: np.ndarray) -> np.ndarray:
        """Calculate matrix inverse"""
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Inverse requires square matrix, got {A.shape}")
        result = np.linalg.inv(A)
        self.operation_history.append("Inverse")
        return result
    
    def rank(self, A: np.ndarray) -> int:
        """Calculate matrix rank"""
        rank = np.linalg.matrix_rank(A)
        self.operation_history.append("Rank")
        return rank
    
    def eigenvalues(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalues and eigenvectors"""
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Eigenvalues require square matrix, got {A.shape}")
        eigenvals, eigenvecs = np.linalg.eig(A)
        self.operation_history.append("Eigenvalues/Eigenvectors")
        return eigenvals, eigenvecs
    
    def trace(self, A: np.ndarray) -> float:
        """Calculate matrix trace"""
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Trace requires square matrix, got {A.shape}")
        trace_val = np.trace(A)
        self.operation_history.append("Trace")
        return trace_val
    
    def power(self, A: np.ndarray, n: int) -> np.ndarray:
        """Raise matrix to power n"""
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix power requires square matrix, got {A.shape}")
        result = np.linalg.matrix_power(A, n)
        self.operation_history.append(f"Power (^{n})")
        return result
    
    def display_menu(self):
        """Display main menu"""
        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + " MAIN MENU ".center(68) + "║")
        print("╠" + "═" * 68 + "╣")
        print("║  BASIC OPERATIONS:                                              ║")
        print("║    1. Matrix Addition          (A + B)                          ║")
        print("║    2. Matrix Subtraction       (A - B)                          ║")
        print("║    3. Matrix Multiplication    (A × B)                          ║")
        print("║    4. Element-wise Multiply    (A ⊙ B)                          ║")
        print("║    5. Scalar Multiplication    (k × A)                          ║")
        print("║                                                                  ║")
        print("║  SINGLE MATRIX OPERATIONS:                                      ║")
        print("║    6. Transpose                (Aᵀ)                             ║")
        print("║    7. Determinant              (|A|)                            ║")
        print("║    8. Inverse                  (A⁻¹)                            ║")
        print("║    9. Rank                                                      ║")
        print("║   10. Eigenvalues & Eigenvectors                                ║")
        print("║   11. Trace                                                     ║")
        print("║   12. Matrix Power             (Aⁿ)                             ║")
        print("║                                                                  ║")
        print("║  UTILITY:                                                       ║")
        print("║   13. View Stored Matrices                                      ║")
        print("║   14. View Operation History                                    ║")
        print("║   15. Clear Storage                                             ║")
        print("║    0. Exit                                                      ║")
        print("╚" + "═" * 68 + "╝")
    
    def run_two_matrix_operation(self, operation_name: str, operation_func):
        """Run operation that requires two matrices"""
        print(f"\n{'═' * 70}")
        print(f"🔢 {operation_name}")
        print(f"{'═' * 70}")
        
        matrix_a = self.input_matrix("Matrix A")
        if matrix_a is None:
            return
        
        self.display_matrix(matrix_a, "Matrix A")
        
        matrix_b = self.input_matrix("Matrix B")
        if matrix_b is None:
            return
        
        self.display_matrix(matrix_b, "Matrix B")
        
        try:
            result = operation_func(matrix_a, matrix_b)
            self.display_matrix(result, f"Result: {operation_name}")
            
            save = input("\n💾 Save result? (y/n): ").strip().lower()
            if save == 'y':
                name = input("Enter name for result: ").strip()
                self.store_matrix(result, name)
            
            print(f"\n✅ {operation_name} completed successfully!")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    def run_single_matrix_operation(self, operation_name: str, operation_func, 
                                    additional_params: dict = None):
        """Run operation that requires one matrix"""
        print(f"\n{'═' * 70}")
        print(f"🔢 {operation_name}")
        print(f"{'═' * 70}")
        
        matrix_a = self.input_matrix("Matrix A")
        if matrix_a is None:
            return
        
        self.display_matrix(matrix_a, "Matrix A")
        
        try:
            if additional_params:
                # Handle operations with additional parameters
                if 'scalar' in additional_params:
                    scalar = float(input("\nEnter scalar value: "))
                    result = operation_func(matrix_a, scalar)
                elif 'power' in additional_params:
                    power = int(input("\nEnter power (integer): "))
                    result = operation_func(matrix_a, power)
                else:
                    result = operation_func(matrix_a)
            else:
                result = operation_func(matrix_a)
            
            # Display result based on type
            if isinstance(result, np.ndarray):
                if result.ndim == 1 and operation_name == "Eigenvalues & Eigenvectors":
                    # Special handling for eigenvalues
                    eigenvals, eigenvecs = self.eigenvalues(matrix_a)
                    print(f"\n{'─' * 70}")
                    print("📊 Eigenvalues:")
                    print(f"{'─' * 70}")
                    for i, val in enumerate(eigenvals):
                        if np.isreal(val):
                            print(f"  λ{i+1} = {val.real:.6f}")
                        else:
                            print(f"  λ{i+1} = {val.real:.6f} + {val.imag:.6f}i")
                    
                    self.display_matrix(eigenvecs, "Eigenvectors")
                    
                    save = input("\n💾 Save eigenvectors? (y/n): ").strip().lower()
                    if save == 'y':
                        name = input("Enter name for eigenvectors: ").strip()
                        self.store_matrix(eigenvecs, name)
                else:
                    self.display_matrix(result, f"Result: {operation_name}")
                    
                    save = input("\n💾 Save result? (y/n): ").strip().lower()
                    if save == 'y':
                        name = input("Enter name for result: ").strip()
                        self.store_matrix(result, name)
            else:
                # Scalar result
                print(f"\n{'─' * 70}")
                print(f"📊 Result: {operation_name}")
                print(f"{'─' * 70}")
                print(f"  {result:.6f}")
                print(f"{'─' * 70}")
            
            print(f"\n✅ {operation_name} completed successfully!")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    def view_storage(self):
        """View all stored matrices"""
        print(f"\n{'═' * 70}")
        print("📦 STORED MATRICES")
        print(f"{'═' * 70}")
        
        if not self.matrix_storage:
            print("No matrices stored yet.")
        else:
            for name, matrix in self.matrix_storage.items():
                print(f"\n🔹 {name}:")
                print(f"   Shape: {matrix.shape}")
                print(f"   Size: {matrix.size} elements")
                print(f"   Data type: {matrix.dtype}")
                
                view = input(f"   View matrix '{name}'? (y/n): ").strip().lower()
                if view == 'y':
                    self.display_matrix(matrix, name)
    
    def view_history(self):
        """View operation history"""
        print(f"\n{'═' * 70}")
        print("📜 OPERATION HISTORY")
        print(f"{'═' * 70}")
        
        if not self.operation_history:
            print("No operations performed yet.")
        else:
            for i, op in enumerate(self.operation_history, 1):
                print(f"  {i}. {op}")
    
    def clear_storage(self):
        """Clear all stored matrices"""
        confirm = input("\n⚠️  Clear all stored matrices? (y/n): ").strip().lower()
        if confirm == 'y':
            self.matrix_storage.clear()
            print("✅ Storage cleared!")
        else:
            print("❌ Cancelled")
    
    def run(self):
        """Main application loop"""
        self.display_header()
        
        while True:
            self.display_menu()
            choice = input("\n👉 Select operation (0-15): ").strip()
            
            try:
                if choice == '1':
                    self.run_two_matrix_operation("Matrix Addition", self.add_matrices)
                elif choice == '2':
                    self.run_two_matrix_operation("Matrix Subtraction", self.subtract_matrices)
                elif choice == '3':
                    self.run_two_matrix_operation("Matrix Multiplication", self.multiply_matrices)
                elif choice == '4':
                    self.run_two_matrix_operation("Element-wise Multiplication", 
                                                  self.element_wise_multiply)
                elif choice == '5':
                    self.run_single_matrix_operation("Scalar Multiplication", 
                                                    self.scalar_multiply, 
                                                    {'scalar': True})
                elif choice == '6':
                    self.run_single_matrix_operation("Transpose", self.transpose)
                elif choice == '7':
                    self.run_single_matrix_operation("Determinant", self.determinant)
                elif choice == '8':
                    self.run_single_matrix_operation("Inverse", self.inverse)
                elif choice == '9':
                    self.run_single_matrix_operation("Rank", self.rank)
                elif choice == '10':
                    self.run_single_matrix_operation("Eigenvalues & Eigenvectors", 
                                                    self.eigenvalues)
                elif choice == '11':
                    self.run_single_matrix_operation("Trace", self.trace)
                elif choice == '12':
                    self.run_single_matrix_operation("Matrix Power", 
                                                    self.power, 
                                                    {'power': True})
                elif choice == '13':
                    self.view_storage()
                elif choice == '14':
                    self.view_history()
                elif choice == '15':
                    self.clear_storage()
                elif choice == '0':
                    print("\n" + "=" * 70)
                    print("Thank you for using Matrix Operations Tool!".center(70))
                    print("=" * 70 + "\n")
                    break
                else:
                    print("\n❌ Invalid choice! Please select 0-15.")
            except KeyboardInterrupt:
                print("\n\n⚠️  Operation cancelled by user.")
                continue
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                continue
            
            input("\n⏎ Press Enter to continue...")


if __name__ == "__main__":
    app = MatrixOperationsTool()
    app.run()
