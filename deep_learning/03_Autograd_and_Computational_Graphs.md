# 3. **Autograd and Computational Graphs**

## Understanding Automatic Differentiation, Backpropagation, and Computation Graphs in PyTorch

PyTorch’s `autograd` is the core engine for automatic differentiation, which plays a crucial role in optimizing deep learning models. This section covers the concepts of automatic differentiation, backpropagation, and computation graphs.

### 3.1 **Automatic Differentiation (Autograd)**

- **Autograd** enables automatic computation of gradients in PyTorch. When performing operations on tensors with `requires_grad=True`, PyTorch tracks the operations and creates a computation graph dynamically.
- During the backward pass, it computes the gradients of each tensor with respect to a loss function using backpropagation.

  Example:
  ```python
  import torch

  # Create a tensor with requires_grad=True to track computations
  x = torch.tensor([2.0, 3.0], requires_grad=True)
  y = x ** 2  # This operation will be tracked in the computation graph

  # Compute the gradient (dy/dx)
  y.sum().backward()
  
  # Print the gradient of x
  print(x.grad)  # Output: tensor([4.0, 6.0])
  ```

### 3.2 **Backpropagation**

- **Backpropagation** is the process used to calculate gradients for all parameters in the neural network during the training process.
- PyTorch uses the computation graph built during the forward pass to calculate the gradients of the loss with respect to each parameter during the backward pass.
  
  Example:
  ```python
  # Define tensors and a simple operation
  a = torch.tensor([2.0, 5.0], requires_grad=True)
  b = a ** 3  # b is now a function of a

  # Perform backpropagation (calculate db/da)
  b.sum().backward()
  
  # Access the gradient of a
  print(a.grad)  # Output: tensor([12.0, 75.0]) - which is the gradient of b with respect to a
  ```

- In this example, `a.grad` stores the derivative of the function `b = a^3` with respect to `a`, showing how PyTorch automates the gradient calculation for optimization.

### 3.3 **Computation Graphs**

- A **computation graph** is a directed graph where nodes represent operations and edges represent tensors flowing between these operations.
- In PyTorch, the graph is created dynamically (also called a **dynamic computation graph** or **define-by-run graph**). This means the graph is built as operations are executed, which is flexible for debugging and creating models that change with input.
  
  Example of dynamic graph creation:
  ```python
  x = torch.tensor(1.0, requires_grad=True)
  y = x ** 2  # Operation is added to the graph
  z = y + 3  # Another operation added to the graph

  z.backward()  # Backpropagate through the graph
  print(x.grad)  # Output: tensor(2.0) - gradient of z with respect to x
  ```

### 3.4 **Using `torch.no_grad()` for Efficiency**

- Sometimes you may want to perform operations without tracking gradients (e.g., during inference). For this, you can use the `torch.no_grad()` context manager, which temporarily disables gradient tracking.
  
  Example:
  ```python
  with torch.no_grad():
      x = torch.tensor([2.0], requires_grad=True)
      y = x ** 2  # No gradients will be computed for this operation

  print(x.grad)  # Will output None since no gradients were tracked
  ```

### 3.5 **Detaching Tensors from the Computation Graph**

- You can also **detach** a tensor from the computation graph if you want to stop tracking operations on it.
  
  Example:
  ```python
  x = torch.tensor([2.0, 3.0], requires_grad=True)
  y = x ** 2

  # Detach y from the graph
  y_detached = y.detach()
  ```

- The detached tensor `y_detached` is no longer connected to the computation graph, meaning further operations on it will not be tracked.

---

