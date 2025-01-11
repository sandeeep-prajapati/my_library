### **Theory: Logistic Regression and Its Role in Binary Classification**

---

#### **Logistic Regression:**

Logistic regression is a statistical model used for binary classification tasks. It is used when the dependent variable is categorical and specifically has two possible outcomes. The primary goal of logistic regression is to model the probability that a given input belongs to a particular class.

The model works by fitting a logistic (sigmoid) function to the linear combination of input features. The output is a probability value between 0 and 1, which can then be thresholded to predict one of the two classes.

The logistic function (also known as the sigmoid function) is given by:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Where:
- \( z \) is the linear combination of input features, typically \( z = w_0 + w_1 x_1 + w_2 x_2 + ... + w_n x_n \), where \( w \) are the model weights and \( x \) are the input features.
- \( \sigma(z) \) outputs a probability value between 0 and 1.

The predicted class is typically determined by applying a threshold:
- If \( \sigma(z) \geq 0.5 \), predict class 1 (positive class).
- If \( \sigma(z) < 0.5 \), predict class 0 (negative class).

---

#### **Role in Binary Classification:**

Logistic regression is specifically designed for binary classification, where the task is to predict one of two classes. It is widely used in various domains such as:
- **Spam detection** (Spam vs. Non-Spam emails)
- **Medical diagnosis** (Diseased vs. Healthy)
- **Sentiment analysis** (Positive vs. Negative sentiment)

In binary classification, logistic regression provides a simple and efficient way to compute probabilities and make predictions. The model is trained using a loss function called **binary cross-entropy** (log loss), which penalizes incorrect predictions based on the distance between the predicted probability and the actual class label.

---

#### **Binary Cross-Entropy Loss:**

The binary cross-entropy loss function for a single prediction is given by:

\[
\text{Loss} = -[y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y})]
\]

Where:
- \( y \) is the true class label (0 or 1).
- \( \hat{y} \) is the predicted probability for class 1 (obtained from the logistic function).

The binary cross-entropy loss is minimized during the training process to improve the model's ability to correctly predict the class label.

---

### **Prompt: Use PyTorch to Classify Emails as Spam or Non-Spam**

Here's how you can implement a logistic regression model using PyTorch to classify emails as spam or non-spam. In this example, we use a simple dataset where emails are represented by features (e.g., frequency of certain words), and the target variable indicates whether the email is spam (1) or not spam (0).

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Dummy dataset for binary classification (Spam vs Non-Spam)
# In real-world cases, you would replace this with actual email feature data
# For simplicity, using digits dataset here as a placeholder for features
data = load_digits()
X = data.data
y = (data.target % 2).astype(int)  # Convert target to binary (even=0, odd=1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Logistic Regression Model (Single-layer Neural Network)
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # One output for binary classification
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Sigmoid activation for binary output

# Initialize the model, loss function, and optimizer
model = LogisticRegressionModel(X_train.shape[1])  # Input dimension = number of features
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the model
with torch.no_grad():
    y_pred = model(X_test_tensor)
    predicted = (y_pred >= 0.5).float()  # Convert probabilities to binary labels
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy on test data: {accuracy * 100:.2f}%')
```

---

### **Explanation of the Code:**

1. **Dataset and Preprocessing:**
   - For simplicity, we used the `load_digits` dataset from `sklearn` to simulate email features. In a real scenario, you would use a dataset with features extracted from emails (e.g., word frequencies, presence of certain keywords, etc.).
   - The target values (`y`) are binary (0 for non-spam and 1 for spam), and we preprocess the features using **StandardScaler** for normalization.

2. **Model Definition:**
   - The model is a simple logistic regression model, implemented using a single linear layer followed by a **sigmoid activation** function to output a probability between 0 and 1.

3. **Loss Function and Optimizer:**
   - The **binary cross-entropy loss** (`BCELoss`) is used for binary classification.
   - **Stochastic Gradient Descent (SGD)** is used to update the model's weights during training.

4. **Training:**
   - The model is trained for 100 epochs. The loss is printed every 10 epochs to track the training progress.

5. **Testing and Evaluation:**
   - After training, the model is evaluated on the test set. The predictions are thresholded at 0.5 (if \( \hat{y} \geq 0.5 \), predict spam), and accuracy is calculated.

---

### **Key Takeaways:**

- **Logistic Regression** is a simple yet powerful algorithm for binary classification problems.
- It outputs probabilities, which can be thresholded to classify data points into two classes.
- The implementation in PyTorch demonstrates how logistic regression can be used for real-world tasks like spam classification, with key components including loss functions and optimizers.
