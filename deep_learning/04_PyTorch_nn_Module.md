# 4. **PyTorch’s `nn` Module**

## Overview of the `torch.nn` Module for Building Neural Networks

The `torch.nn` module in PyTorch provides tools to build and train neural networks. It contains layers, loss functions, and other building blocks necessary to define and train deep learning models. This section gives an overview of how to use the `nn` module to build neural networks.

### 4.1 **What is the `torch.nn` Module?**

- The `torch.nn` module abstracts much of the complexity of neural networks. It provides pre-built components like layers, activation functions, and loss functions, making it easy to assemble a neural network.
- Key components:
  - **`nn.Module`:** The base class for all neural network layers.
  - **Layers:** Predefined layers like fully connected layers (`nn.Linear`), convolutional layers (`nn.Conv2d`), etc.
  - **Activation functions:** Non-linear functions such as `ReLU`, `Sigmoid`, etc.
  - **Loss functions:** Methods for calculating error during training (e.g., `nn.CrossEntropyLoss`, `nn.MSELoss`).

### 4.2 **Defining a Neural Network with `nn.Module`**

- To define a neural network, you subclass `nn.Module` and define the layers in the `__init__` method. The forward pass (computation) is defined in the `forward()` method.

  Example:
  ```python
  import torch
  import torch.nn as nn
  
  class SimpleNN(nn.Module):
      def __init__(self):
          super(SimpleNN, self).__init__()
          # Define layers
          self.fc1 = nn.Linear(10, 5)  # Fully connected layer: 10 input features, 5 output features
          self.fc2 = nn.Linear(5, 2)   # Another fully connected layer

      def forward(self, x):
          # Define forward pass
          x = self.fc1(x)  # Pass input through the first layer
          x = torch.relu(x)  # Apply ReLU activation
          x = self.fc2(x)  # Pass through second layer
          return x
  
  # Instantiate and test the network
  model = SimpleNN()
  input_tensor = torch.randn(1, 10)  # Create a random input tensor
  output = model(input_tensor)  # Forward pass through the model
  print(output)
  ```

### 4.3 **Common Layers in `torch.nn`**

- **`nn.Linear`:** A fully connected layer (also called dense layer) that applies a linear transformation to the input.  
  Example:
  ```python
  layer = nn.Linear(10, 5)  # 10 input features, 5 output features
  ```

- **`nn.Conv2d`:** A 2D convolutional layer commonly used in image data processing.  
  Example:
  ```python
  conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)  # 3 input channels, 16 output channels
  ```

- **`nn.ReLU`:** The Rectified Linear Unit (ReLU) is a commonly used activation function.  
  Example:
  ```python
  activation = nn.ReLU()
  ```

### 4.4 **Loss Functions in `torch.nn`**

- PyTorch provides a variety of loss functions to compute the error between the predicted output and the target.
  - **`nn.CrossEntropyLoss`:** Commonly used for classification tasks where the target is a class label.
  - **`nn.MSELoss`:** Used for regression tasks where the target is a continuous value.

  Example:
  ```python
  loss_fn = nn.CrossEntropyLoss()  # For classification
  ```

### 4.5 **Optimizing the Model**

- To optimize a model, you combine the `nn.Module` with PyTorch’s optimization functions, usually from the `torch.optim` package. The optimizer updates the weights during training based on the gradients computed by backpropagation.

  Example:
  ```python
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer
  ```

### 4.6 **Training Loop**

- The typical training loop in PyTorch involves:
  1. Forward pass: Compute the predictions.
  2. Compute loss: Use a loss function to calculate error.
  3. Backward pass: Perform backpropagation using `loss.backward()`.
  4. Update weights: Use the optimizer to adjust the model's parameters.

  Example of a basic training loop:
  ```python
  for epoch in range(100):
      optimizer.zero_grad()  # Zero the gradients
      output = model(input_tensor)  # Forward pass
      loss = loss_fn(output, target)  # Compute loss
      loss.backward()  # Backpropagation
      optimizer.step()  # Update weights
  ```

---

