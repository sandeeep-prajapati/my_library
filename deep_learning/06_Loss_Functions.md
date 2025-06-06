# 6. **Loss Functions**

## Understanding and Implementing Various Loss Functions such as Cross-Entropy, MSE, and MAE

Loss functions play a crucial role in training machine learning models by quantifying the difference between predicted outputs and actual targets. During the optimization process, the goal is to minimize this loss, which guides the model to make better predictions. Here we explore several commonly used loss functions in PyTorch, including **Cross-Entropy**, **Mean Squared Error (MSE)**, and **Mean Absolute Error (MAE)**.

### 6.1 **Cross-Entropy Loss**

- **Cross-Entropy Loss** is used primarily for classification tasks. It measures the difference between two probability distributions: the true distribution (actual labels) and the predicted distribution (model’s predictions). It's especially effective in multi-class classification problems.

- **Formula**:
  \[
  L = - \sum_{i} y_i \log(\hat{y_i})
  \]
  Where:
  - \( y_i \) is the actual class label (as one-hot encoded vectors),
  - \( \hat{y_i} \) is the predicted probability for each class.

- **In PyTorch**: The `nn.CrossEntropyLoss()` function is commonly used for this. It combines softmax and negative log-likelihood in one step.

  **Example**:
  ```python
  loss_fn = torch.nn.CrossEntropyLoss()
  predictions = model(input_tensor)
  loss = loss_fn(predictions, target)
  ```

- **Use Case**: Ideal for tasks like image classification, text classification, and any task where the goal is to assign a sample to one of many discrete categories.

### 6.2 **Mean Squared Error (MSE) Loss**

- **Mean Squared Error (MSE)** is commonly used in regression tasks. It calculates the average of the squares of the errors—the differences between the predicted values and the actual target values. The squared nature of this loss makes it sensitive to outliers (larger errors get amplified).

- **Formula**:
  \[
  L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
  \]
  Where:
  - \( y_i \) is the actual target value,
  - \( \hat{y_i} \) is the predicted value,
  - \( n \) is the number of samples.

- **In PyTorch**: Use `nn.MSELoss()` to implement this.

  **Example**:
  ```python
  loss_fn = torch.nn.MSELoss()
  predictions = model(input_tensor)
  loss = loss_fn(predictions, target)
  ```

- **Use Case**: Suitable for regression problems like predicting continuous values (e.g., stock prices, housing prices).

### 6.3 **Mean Absolute Error (MAE) Loss**

- **Mean Absolute Error (MAE)**, also known as **L1 Loss**, calculates the average of the absolute differences between the predicted values and actual target values. Unlike MSE, MAE is more robust to outliers since it doesn’t square the differences.

- **Formula**:
  \[
  L = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}|
  \]

- **In PyTorch**: Implement MAE with `nn.L1Loss()`.

  **Example**:
  ```python
  loss_fn = torch.nn.L1Loss()
  predictions = model(input_tensor)
  loss = loss_fn(predictions, target)
  ```

- **Use Case**: Often used in regression tasks where robustness to outliers is important, as MAE is less sensitive to large differences than MSE.

### 6.4 **Choosing the Right Loss Function**

- **Cross-Entropy Loss**: Use for classification problems where the output is a probability distribution over multiple classes.
  
- **MSE Loss**: Best for regression tasks where you predict continuous values, and large errors need to be penalized more heavily.

- **MAE Loss**: Good for regression tasks where outliers are present, and you don’t want the model to be overly sensitive to them.

---

