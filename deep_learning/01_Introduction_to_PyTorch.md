# 1. **Introduction to PyTorch**

## Overview of PyTorch’s Core Features

PyTorch is an open-source machine learning library widely used for deep learning applications. It provides flexibility and speed in building, training, and deploying machine learning models. Below are PyTorch's core features that set it apart:

### 1.1 **Tensors**

- **Tensors** are the fundamental data structure in PyTorch, similar to NumPy arrays but with added functionality to support operations on GPUs.
- Tensors can handle multi-dimensional data and perform fast matrix operations.
- Key points about tensors:
  - **Initialization:** You can create tensors from lists, NumPy arrays, or random values.
    ```python
    import torch
    x = torch.tensor([1.0, 2.0, 3.0])
    ```
  - **Operations:** PyTorch supports a wide range of tensor operations (addition, multiplication, etc.).
    ```python
    y = torch.tensor([4.0, 5.0, 6.0])
    z = x + y  # Element-wise addition
    ```
  - **GPU support:** You can easily move tensors to and from the GPU for faster computation.
    ```python
    x_gpu = x.to("cuda")  # Move tensor to GPU
    ```

### 1.2 **Autograd (Automatic Differentiation)**

- **Autograd** is PyTorch’s automatic differentiation engine that powers neural network training by computing gradients for backpropagation.
- It enables automatic computation of the gradients required to optimize model parameters.
- Key points about autograd:
  - **Tracking gradients:** PyTorch tracks tensor operations when `requires_grad=True`, allowing it to compute derivatives for optimization.
    ```python
    a = torch.tensor([2.0, 3.0], requires_grad=True)
    b = a ** 2
    b.backward(torch.tensor([1.0, 1.0]))  # Compute gradients
    print(a.grad)  # Output: tensor([4.0, 6.0])
    ```
  - **Dynamic computation graphs:** PyTorch constructs the computation graph dynamically as operations are performed, making it flexible for model building and debugging.

### 1.3 **Dynamic Computation Graphs**

- Unlike static computation graphs (as in TensorFlow), PyTorch uses **dynamic computation graphs**.
- This means that the graph is built on-the-fly, allowing for more intuitive code execution, especially useful when:
  - The network architecture is changing dynamically.
  - You need to run different computations based on the input data.
  
  Example:
  ```python
  x = torch.randn(3, requires_grad=True)
  if x.sum() > 0:
      y = x * 2
  else:
      y = x / 2
  y.sum().backward()
  ```

- Benefits of dynamic graphs:
  - **Flexibility:** You can adjust the network during runtime.
  - **Ease of debugging:** The graph is defined by the flow of operations, making debugging easier in an imperative style.

---

