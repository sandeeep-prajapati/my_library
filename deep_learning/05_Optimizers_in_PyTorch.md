# 5. **Optimizers in PyTorch**

## Popular Optimization Algorithms: SGD, Adam, RMSprop, and How to Use Them with `torch.optim`

Optimizers in PyTorch are essential for training neural networks. They adjust the model’s parameters (weights) based on the gradients computed during backpropagation to minimize the loss function. PyTorch provides several popular optimization algorithms through the `torch.optim` module. This section introduces some widely-used optimizers and how to use them.

### 5.1 **Stochastic Gradient Descent (SGD)**

- **Stochastic Gradient Descent (SGD)** is one of the simplest optimization algorithms. It updates the model parameters based on the gradient of the loss with respect to the parameters, scaled by a learning rate.
- Formula:  
  \[
  \theta = \theta - \eta \nabla J(\theta)
  \]
  Where:
  - \( \theta \) are the model parameters (weights),
  - \( \eta \) is the learning rate,
  - \( \nabla J(\theta) \) is the gradient of the loss function.

- **Mini-batch SGD** is commonly used in practice, where the update is performed over a small batch of data points rather than the entire dataset.

  Example:
  ```python
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer
  ```

- **Momentum** can be added to SGD to accelerate convergence and avoid local minima:
  ```python
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  ```

### 5.2 **Adam (Adaptive Moment Estimation)**

- **Adam** is one of the most popular and effective optimizers. It combines the benefits of **Adagrad** and **RMSprop**, making it suitable for sparse gradients and noisy data.
- Adam maintains two moving averages for each parameter:
  1. **First moment** (mean of gradients).
  2. **Second moment** (mean of squared gradients).

  The parameters are updated using these moments, which helps the optimizer adapt the learning rate for each parameter.

  Example:
  ```python
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
  ```

- **Advantages of Adam**:
  - Works well with large datasets.
  - Efficient with sparse gradients.
  - Suitable for non-stationary objectives.

### 5.3 **RMSprop (Root Mean Square Propagation)**

- **RMSprop** is another optimizer designed to handle the problem of vanishing learning rates during training. It normalizes the gradients by dividing them by a moving average of their magnitudes, ensuring the learning rate doesn’t decay too quickly.
  
  Formula for parameter update:
  \[
  \theta = \theta - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \nabla J(\theta)
  \]
  Where:
  - \( E[g^2]_t \) is the exponentially weighted average of squared gradients,
  - \( \epsilon \) is a small value to prevent division by zero.

  Example:
  ```python
  optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)  # RMSprop optimizer
  ```

- **RMSprop** is typically used in recurrent neural networks and works well in practice for many deep learning models.

### 5.4 **Using Optimizers in PyTorch**

1. **Initialize the optimizer** with the model’s parameters:
   ```python
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
   ```

2. **In each training iteration**:
   - Zero out the gradients with `optimizer.zero_grad()`. This prevents accumulation of gradients from previous iterations.
   - Perform backpropagation using `loss.backward()` to compute gradients.
   - Update the model’s parameters using `optimizer.step()`.

   Example training loop:
   ```python
   for epoch in range(epochs):
       optimizer.zero_grad()  # Reset gradients
       outputs = model(inputs)  # Forward pass
       loss = loss_fn(outputs, targets)  # Compute loss
       loss.backward()  # Backpropagate
       optimizer.step()  # Update parameters
   ```

### 5.5 **Choosing the Right Optimizer**

- **SGD**: Works well for simple tasks or when using techniques like momentum and learning rate decay. It is typically more stable and can lead to better generalization.
  
- **Adam**: Generally works well out-of-the-box for most tasks, especially for more complex models and larger datasets. Adam is faster and requires less tuning, but can overfit without proper regularization.

- **RMSprop**: A good alternative to Adam, especially for recurrent neural networks (RNNs) and other models where the learning rate needs to be adjusted dynamically.

---

