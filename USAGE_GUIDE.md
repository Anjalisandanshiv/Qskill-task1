# Matrix Operations Tool - Usage Guide

## 🚀 Getting Started

### Option 1: Command-Line Interface (CLI)
```bash
python matrix_operations_cli.py
```

**Features:**
- Interactive menu system
- Multiple matrix input methods
- Store and retrieve matrices
- Operation history tracking
- Formatted console output

### Option 2: Web Interface
```bash
# Open in your browser
firefox matrix_operations_web.html
# or simply double-click the file
```

**Features:**
- Modern, responsive design
- Real-time validation
- Visual matrix display
- Quick example loading
- One-click operations

### Option 3: Python Script Integration
```python
import numpy as np

# Your code here using NumPy
```

---

## 📝 Input Methods

### CLI Input Methods

#### Method 1: Manual Entry (Row by Row)
```
Enter number of rows: 3
Enter number of columns: 3

Row 1: 1 2 3
Row 2: 4 5 6
Row 3: 7 8 9
```

#### Method 2: MATLAB-Style String
```
Enter matrix (e.g., '1 2 3; 4 5 6; 7 8 9'):
→ 1 2 3; 4 5 6; 7 8 9
```

#### Method 3: Generate Matrices
- **Random**: Random integers in specified range
- **Identity**: Square identity matrix (I)
- **Zero**: All-zero matrix

### Web Interface Input

#### Direct Entry
```
Rows: 3    Cols: 3

Matrix values:
1 2 3
4 5 6
7 8 9
```

#### Quick Examples
Click buttons:
- [Identity] → Creates identity matrix
- [Random] → Generates random values (1-9)
- [Zero] → Creates zero matrix

---

## 🎯 Operation Examples

### Example 1: Matrix Addition
**Problem:** Add two 2×2 matrices

**Input:**
```
Matrix A:        Matrix B:
[1  2]          [5  6]
[3  4]          [7  8]
```

**Steps (CLI):**
1. Select option 1 (Matrix Addition)
2. Enter Matrix A: `1 2; 3 4`
3. Enter Matrix B: `5 6; 7 8`

**Steps (Web):**
1. Enter A values: `1 2` and `3 4`
2. Enter B values: `5 6` and `7 8`
3. Click "➕ Add (A + B)"

**Result:**
```
[6   8]
[10  12]
```

---

### Example 2: Matrix Multiplication
**Problem:** Multiply two 2×2 matrices

**Input:**
```
Matrix A:        Matrix B:
[1  2]          [5  6]
[3  4]          [7  8]
```

**Calculation:**
```
Result[0,0] = (1×5) + (2×7) = 5 + 14 = 19
Result[0,1] = (1×6) + (2×8) = 6 + 16 = 22
Result[1,0] = (3×5) + (4×7) = 15 + 28 = 43
Result[1,1] = (3×6) + (4×8) = 18 + 32 = 50
```

**Result:**
```
[19  22]
[43  50]
```

---

### Example 3: Finding Inverse
**Problem:** Find the inverse of a 2×2 matrix

**Input:**
```
Matrix C:
[2  -1]
[1   3]
```

**Steps:**
1. Calculate determinant: |C| = (2×3) - (-1×1) = 7
2. Since det ≠ 0, inverse exists
3. Apply inverse formula

**Result:**
```
C⁻¹:
[ 0.4286   0.1429]
[-0.1429   0.2857]
```

**Verification:** C × C⁻¹ = I
```
[1  0]
[0  1]
```

---

### Example 4: Eigenvalues
**Problem:** Find eigenvalues of a matrix

**Input:**
```
Matrix A:
[4  -2]
[1   1]
```

**Steps:**
1. Solve characteristic equation: det(A - λI) = 0
2. Find eigenvalues

**Result:**
```
λ₁ = 3.000
λ₂ = 2.000

Eigenvectors:
v₁ = [0.8944]    v₂ = [0.7071]
     [0.4472]         [0.7071]
```

---

## 🔧 Common Operations Guide

### Basic Operations

#### Addition (A + B)
- **Requirement:** Same dimensions
- **Formula:** C[i,j] = A[i,j] + B[i,j]
- **Example:** [1,2] + [3,4] = [4,6]

#### Subtraction (A - B)
- **Requirement:** Same dimensions
- **Formula:** C[i,j] = A[i,j] - B[i,j]
- **Example:** [5,6] - [2,1] = [3,5]

#### Multiplication (A × B)
- **Requirement:** A.cols = B.rows
- **Formula:** C[i,j] = Σ A[i,k] × B[k,j]
- **Example:** [2,3] × [1;2] = [8]

### Advanced Operations

#### Transpose (Aᵀ)
- **Effect:** Swap rows ↔ columns
- **Example:**
  ```
  [1 2 3]ᵀ    [1 4]
  [4 5 6]  =  [2 5]
              [3 6]
  ```

#### Determinant (|A|)
- **Use:** Check if inverse exists
- **2×2 Formula:** ad - bc
- **Example:** |[2,-1; 1,3]| = 6-(-1) = 7

#### Inverse (A⁻¹)
- **Requirement:** det(A) ≠ 0
- **Property:** A × A⁻¹ = I
- **Use:** Solving linear systems

---

## 💡 Tips & Tricks

### Input Tips
1. **Use spaces** to separate values in rows
2. **Use semicolons (;)** to separate rows (MATLAB style)
3. **Validate first** before complex operations
4. **Store results** for reuse

### Operation Tips
1. **Check dimensions** before multiplication
2. **Verify determinant** before computing inverse
3. **Use identity** to test operations
4. **Compare with hand calculations** for small matrices

### Performance Tips
1. **Start small** - Test with 2×2 or 3×3 matrices
2. **Use random** for quick testing
3. **Store frequently used** matrices
4. **Leverage NumPy** for efficiency

---

## 🎓 Learning Path

### Beginner
1. Start with addition and subtraction
2. Try transpose operations
3. Practice with small matrices (2×2)
4. Verify results by hand

### Intermediate
1. Learn matrix multiplication rules
2. Understand determinants
3. Calculate inverses
4. Explore eigenvalues

### Advanced
1. Matrix powers and functions
2. Singular value decomposition
3. Matrix norms
4. Numerical stability

---

## 🐛 Troubleshooting

### Error: "Matrix dimensions don't match"
**Solution:** For addition/subtraction, matrices must be same size.

**Example:**
```
✗ [2×3] + [3×2]  ← Different dimensions
✓ [2×3] + [2×3]  ← Same dimensions
```

### Error: "Cannot multiply matrices"
**Solution:** Inner dimensions must match (A.cols = B.rows).

**Example:**
```
✗ [2×3] × [2×4]  ← Inner dimensions 3≠2
✓ [2×3] × [3×4]  ← Inner dimensions 3=3 → Result: 2×4
```

### Error: "Matrix is singular"
**Solution:** Determinant is zero, no inverse exists.

**Example:**
```
A = [1 2]  |A| = 2-2 = 0  (singular)
    [1 2]
```

### Error: "Invalid input format"
**Solution:** Check spacing and row consistency.

**Correct:**
```
1 2 3
4 5 6
```

**Incorrect:**
```
1 2  3
4 5 6 7  ← Inconsistent columns
```

---

## 📊 Quick Reference

### Matrix Dimensions
| Matrix | Dimensions | Size |
|--------|-----------|------|
| Vector | n×1 or 1×n | n elements |
| Square | n×n | n² elements |
| Rectangular | m×n | m·n elements |

### Operation Compatibility
| Operation | Requirement | Result |
|-----------|-------------|--------|
| A + B | Same size | Same size |
| A - B | Same size | Same size |
| A × B | A.cols = B.rows | A.rows × B.cols |
| A ⊙ B | Same size | Same size |
| Aᵀ | Any | Transposed |
| A⁻¹ | Square, det≠0 | Same size |

### Special Matrices
| Type | Description | Example |
|------|-------------|---------|
| Identity (I) | Diagonal = 1 | [1 0; 0 1] |
| Zero (0) | All zeros | [0 0; 0 0] |
| Diagonal | Off-diagonal = 0 | [a 0; 0 d] |
| Symmetric | A = Aᵀ | [a b; b c] |

---

## 🎯 Practice Exercises

### Exercise 1: Basic Operations
```
Given: A = [1 2]    B = [5 6]
           [3 4]        [7 8]

Calculate:
a) A + B
b) A - B
c) A × B
d) 3A
e) Aᵀ
```

### Exercise 2: Inverse & Verification
```
Given: C = [2 -1]
           [1  3]

Calculate:
a) |C| (determinant)
b) C⁻¹ (inverse)
c) Verify: C × C⁻¹ = I
```

### Exercise 3: Properties
```
Given: A = [1 2]    I = [1 0]
           [3 4]        [0 1]

Verify:
a) A × I = A
b) (Aᵀ)ᵀ = A
c) |A| ≠ 0 ⟹ A⁻¹ exists
```

---

## 📚 Additional Resources

### Documentation
- **NumPy Docs:** https://numpy.org/doc/
- **Linear Algebra:** Khan Academy
- **Matrix Theory:** Gilbert Strang's textbook

### Video Tutorials
- 3Blue1Brown - Essence of Linear Algebra
- MIT OpenCourseWare - 18.06
- Khan Academy - Linear Algebra playlist

### Practice Platforms
- Wolfram Alpha (verification)
- MATLAB Online (experimentation)
- Symbolab (step-by-step solutions)

---

**Happy Learning! 🎓**

For more help, refer to README.md or run the demo script!
