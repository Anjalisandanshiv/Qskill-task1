# Matrix Operations Tool 🔢

## 🎯 Overview
A comprehensive matrix operations tool built with Python and NumPy, featuring both command-line and web-based interfaces. Perform various matrix operations including addition, subtraction, multiplication, transpose, determinant calculation, and many more advanced operations.

## ✨ Key Features

### Basic Operations
- ➕ Matrix Addition
- ➖ Matrix Subtraction  
- ✖️ Matrix Multiplication
- ⊙ Element-wise Multiplication (Hadamard product)
- 🔢 Scalar Multiplication

### Advanced Operations
- ↔️ Transpose
- |A| Determinant
- A⁻¹ Inverse
- 📊 Rank
- Tr Trace
- Aⁿ Matrix Power
- λ Eigenvalues & Eigenvectors

### Utility Features
- 💾 Matrix storage system
- 📜 Operation history tracking
- 🎲 Random matrix generation
- ✓ Automatic validation
- 🔄 Matrix manipulation tools

## 📦 Files Included

1. **matrix_operations_cli.py** - Full-featured command-line interface
2. **matrix_operations_web.html** - Interactive web-based interface
3. **matrix_demo.py** - Comprehensive demonstration with examples
4. **README.md** - Complete documentation

## 🚀 Quick Start

### Command-Line Interface
```bash
python matrix_operations_cli.py
```

### Web Interface
Open `matrix_operations_web.html` in any modern browser.

### Demo Script
```bash
python matrix_demo.py
```

## 💻 Usage Examples

### CLI Example
```python
# Interactive menu-driven interface
# Select operation -> Enter matrices -> View results
# Multiple input methods available:
#   - Manual entry (row by row)
#   - MATLAB-style string ("1 2 3; 4 5 6")
#   - Generate random, identity, or zero matrices
```

### Web Interface Example
1. Enter matrix dimensions (rows × columns)
2. Input values (space-separated)
3. Click operation button
4. View formatted results instantly

### Python Code Example
```python
import numpy as np

# Create matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Perform operations
sum_result = A + B           # Addition
product = np.matmul(A, B)    # Multiplication
det = np.linalg.det(A)       # Determinant
inv = np.linalg.inv(A)       # Inverse
transpose = A.T              # Transpose

print(f"A + B =\n{sum_result}")
print(f"Determinant: {det}")
```

## 📊 Supported Operations

| Operation | Requirements | Description |
|-----------|--------------|-------------|
| Addition | Same dimensions | Element-wise sum |
| Subtraction | Same dimensions | Element-wise difference |
| Multiplication | A.cols = B.rows | Matrix product |
| Element-wise | Same dimensions | Hadamard product |
| Transpose | None | Swap rows/columns |
| Determinant | Square matrix | Scalar value |
| Inverse | Square, det ≠ 0 | A⁻¹ such that AA⁻¹ = I |
| Trace | Square matrix | Sum of diagonal |
| Rank | None | Linear independence |
| Power | Square matrix | A multiplied by itself n times |

## 🎨 Interface Features

### CLI Features
- Interactive menu system
- Multiple input methods
- Formatted matrix display
- Operation history tracking
- Matrix storage and retrieval
- Error handling with helpful messages

### Web Features
- Modern, responsive design
- Real-time validation
- Quick example loading
- Matrix swap functionality
- Color-coded results
- Formatted matrix visualization

## 📝 Example Operations

### Matrix Addition
```
A = [1 2]    B = [5 6]    A + B = [6  8]
    [3 4]        [7 8]            [10 12]
```

### Matrix Multiplication
```
A = [1 2]    B = [5 6]    A × B = [19 22]
    [3 4]        [7 8]            [43 50]
```

### Determinant & Inverse
```
C = [2 -1]      |C| = 7
    [1  3]

C⁻¹ = [ 0.429  0.143]     C × C⁻¹ = [1 0]  (Identity)
      [-0.143  0.286]               [0 1]
```

## 🔧 Requirements

- Python 3.7 or higher
- NumPy library

### Installation
```bash
pip install numpy --break-system-packages
```

## 💡 Use Cases

- **Educational**: Learn matrix operations interactively
- **Professional**: Quick calculations and verification
- **Research**: Linear algebra experiments and prototyping
- **Development**: Test algorithms and transformations

## 🎯 Tips

1. **Validate First**: Check matrix dimensions before operations
2. **Store Results**: Save intermediate results for complex calculations
3. **Use Examples**: Load identity/random matrices for testing
4. **Check Determinant**: Verify det ≠ 0 before computing inverse

## 📚 Mathematical Properties

- **(AB)C = A(BC)** - Associative
- **A + B = B + A** - Commutative (addition)
- **(A + B)ᵀ = Aᵀ + Bᵀ** - Transpose distribution
- **(AB)ᵀ = BᵀAᵀ** - Transpose of product
- **|AB| = |A||B|** - Determinant of product

## 🐛 Common Errors & Solutions

### Dimension Mismatch
```
Error: Matrix dimensions don't match
Solution: Ensure matrices have compatible dimensions
```

### Singular Matrix
```
Error: Matrix is singular (determinant = 0)
Solution: Matrix has no inverse
```

### Invalid Input
```
Error: Expected N values, got M
Solution: Check row values match column count
```

## 📈 Performance

| Operation | Time Complexity |
|-----------|----------------|
| Addition | O(n²) |
| Multiplication | O(n³) |
| Determinant | O(n³) |
| Inverse | O(n³) |
| Transpose | O(n²) |

## 🎓 Learn More

- NumPy Documentation: https://numpy.org/doc/
- Linear Algebra Resources: Khan Academy, MIT OCW
- Matrix Theory: Strang's "Linear Algebra"

---

**Created for educational and practical matrix operations**  
**Version**: 1.0 | **Updated**: February 2026

**Happy Computing! 🎉**
