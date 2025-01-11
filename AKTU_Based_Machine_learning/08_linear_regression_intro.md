### **Theory: Linear Regression and Its Assumptions**

---

#### **Linear Regression:**
Linear regression is one of the most basic and widely used techniques for predicting a continuous target variable based on one or more features. The goal of linear regression is to fit a line (or hyperplane in higher dimensions) that best represents the relationship between the input features and the output target variable.

The model assumes that there is a linear relationship between the input variables and the output variable. The general equation for a linear regression model is:

\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon
\]

Where:
- \( y \) is the dependent variable (target).
- \( x_1, x_2, \dots, x_n \) are the independent variables (features).
- \( \beta_0 \) is the intercept.
- \( \beta_1, \beta_2, \dots, \beta_n \) are the coefficients (weights) of the features.
- \( \epsilon \) is the error term or residual, which accounts for the difference between the observed value and the predicted value.

---

#### **Assumptions of Linear Regression:**

1. **Linearity:**  
   The relationship between the independent and dependent variables is assumed to be linear. This means that the change in the target variable is proportional to the changes in the input features.

2. **Independence of Errors:**  
   The residuals (errors) should be independent of each other. There should be no correlation between the residuals. This assumption is important to avoid issues like autocorrelation, which can lead to unreliable coefficient estimates.

3. **Homoscedasticity:**  
   The variance of the residuals should be constant across all levels of the independent variables. In other words, the spread of residuals should not change as the value of the independent variables increases. This is crucial for valid statistical inference.

4. **Normality of Errors:**  
   The errors (residuals) should be normally distributed. This assumption is important when making statistical inferences about the coefficients (such as confidence intervals or hypothesis testing).

5. **No Multicollinearity:**  
   The independent variables should not be highly correlated with each other. Multicollinearity can make the model unstable and lead to inflated standard errors for the coefficients.

---

### **Prompt: Implement Linear Regression in PyTorch for Predicting Housing Prices**

Here is a PyTorch implementation of linear regression for predicting housing prices based on the number of rooms and square footage.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Example housing data (number of rooms and square footage)
# Features: [rooms, sqft]
# Target: [price]
X = np.array([[3, 1500], [4, 1800], [3, 1300], [5, 2200], [4, 1600]], dtype=np.float32)
y = np.array([400000, 500000, 350000, 600000, 450000], dtype=np.float32)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).view(-1, 1)  # Reshape target to column vector

# Define the model (Linear Regression: y = wx + b)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # 2 input features (rooms, sqft), 1 output (price)

    def forward(self, x):
        return self.linear(x)

# Initialize the model
model = LinearRegressionModel()

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training the model
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    predictions = model(X_tensor)

    # Compute the loss
    loss = criterion(predictions, y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Get the learned parameters (weights and bias)
weights, bias = model.linear.parameters()
print(f"Learned weights: {weights[0][0].item():.4f}, {weights[0][1].item():.4f}")
print(f"Learned bias: {bias.item():.4f}")

# Make predictions on the training data
predictions = model(X_tensor).detach().numpy()

# Plot the predictions vs actual prices
plt.scatter(X[:, 0], y, color='blue', label='Actual Prices')  # Actual prices (scatter plot)
plt.scatter(X[:, 0], predictions, color='red', label='Predicted Prices')  # Predicted prices (red dots)
plt.xlabel('Number of Rooms')
plt.ylabel('Price')
plt.legend()
plt.title('Housing Price Prediction')
plt.show()

# Making a new prediction for a house with 4 rooms and 1600 sqft
new_house = np.array([[4, 1600]], dtype=np.float32)
new_house_tensor = torch.tensor(new_house)
predicted_price = model(new_house_tensor).item()
print(f"Predicted price for 4 rooms and 1600 sqft: ${predicted_price:.2f}")
```

---

### **Explanation of the Code:**

1. **Dataset:**  
   - The dataset contains features like the number of rooms and square footage (`X`) and the corresponding housing prices (`y`).

2. **Model Definition:**  
   - A simple linear regression model is defined using PyTorch's `nn.Linear` module, where `2` represents the number of features (rooms and square footage) and `1` represents the target (price).

3. **Training:**  
   - The model is trained for 1000 epochs using the Mean Squared Error (MSE) loss function and Stochastic Gradient Descent (SGD) optimizer. The model learns by updating its parameters to minimize the error between predicted and actual prices.

4. **Prediction and Visualization:**  
   - After training, the model is used to predict housing prices, and the results are visualized in a scatter plot comparing actual vs. predicted prices based on the number of rooms.

5. **New Prediction:**  
   - The model is also used to predict the price of a new house with 4 rooms and 1600 square feet.

---

### **Key Takeaways:**
- Linear regression is a fundamental technique for predictive modeling where the target variable is continuous.
- PyTorch makes it easy to implement and train linear regression models using simple layers and optimization routines.
- The assumptions of linear regression, such as linearity and independence of errors, must be verified for the model to produce reliable results.