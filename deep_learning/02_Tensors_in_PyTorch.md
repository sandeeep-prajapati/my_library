# 2. **Tensors in PyTorch**

## Tensor Operations, Indexing, Slicing, Broadcasting, and Type Conversions

Tensors are a core feature of PyTorch and enable a wide range of operations for machine learning and numerical computation. This section provides a detailed overview of tensor operations, indexing, slicing, broadcasting, and type conversions.

### 2.1 **Tensor Operations**

- PyTorch tensors support various element-wise and matrix operations such as addition, multiplication, and linear algebra operations.
  
  Example:
  ```python
  import torch
  a = torch.tensor([1.0, 2.0, 3.0])
  b = torch.tensor([4.0, 5.0, 6.0])
  
  # Element-wise addition
  c = a + b  # Output: tensor([5.0, 7.0, 9.0])

  # Matrix multiplication (dot product)
  d = torch.matmul(a, b)  # Output: tensor(32.0)
  ```

### 2.2 **Indexing**

- PyTorch tensors support standard Python-style indexing, allowing easy access to specific elements or ranges.
  
  Example:
  ```python
  x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  
  # Accessing specific elements
  print(x[0, 2])  # Output: tensor(3)

  # Accessing entire row or column
  print(x[1, :])  # Output: tensor([4, 5, 6])
  ```

### 2.3 **Slicing**

- You can extract a range of values from a tensor using slicing, which follows the `start:end:step` notation.
  
  Example:
  ```python
  x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
  
  # Slice elements from index 2 to 5
  print(x[2:6])  # Output: tensor([2, 3, 4, 5])
  
  # Slice with step size
  print(x[::2])  # Output: tensor([0, 2, 4, 6])
  ```

### 2.4 **Broadcasting**

- **Broadcasting** is a mechanism that automatically expands the dimensions of smaller tensors to match the shape of larger tensors during operations.
  
  Example:
  ```python
  a = torch.tensor([1.0, 2.0, 3.0])
  b = torch.tensor([[1.0], [2.0], [3.0]])
  
  # Broadcasting allows the tensors to be added despite shape differences
  c = a + b  # Output: tensor([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])
  ```

- **Rules for Broadcasting:**
  - If the dimensions do not match, PyTorch automatically expands the smaller tensor to the shape of the larger one.
  - If a dimension is 1, it can be broadcasted to match the other tensor’s dimension.

### 2.5 **Type Conversions**

- PyTorch tensors can be converted between different types (e.g., from integer to float) using the `type()` or `to()` methods.
  
  Example:
  ```python
  x = torch.tensor([1, 2, 3], dtype=torch.int32)
  
  # Convert to float
  x_float = x.float()  # Output: tensor([1.0, 2.0, 3.0])

  # Convert to a specific device (e.g., GPU)
  x_gpu = x.to("cuda")
  ```

- Common tensor types in PyTorch:
  - `torch.float32` or `torch.float`: 32-bit floating point
  - `torch.int64` or `torch.long`: 64-bit integer
  - `torch.bool`: Boolean

---

