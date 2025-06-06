Here’s an example of how to implement a simple spam detection model using **PyTorch** with `nn.Linear` and `nn.CrossEntropyLoss`. We'll use a fully connected neural network to classify whether a message is spam or ham.

### 1. **Steps in the PyTorch Pipeline:**
1. Preprocess the data.
2. Convert the text to numerical features (BoW/TF-IDF).
3. Define the model using `nn.Linear` layers.
4. Train the model with `nn.CrossEntropyLoss`.
5. Evaluate the model.

---

### 2. **Spam Detection using PyTorch:**

#### Step 1: **Data Preprocessing**
Before feeding data into the PyTorch model, we need to vectorize the text using something like TF-IDF or BoW. In this example, we’ll use the `CountVectorizer` to convert text into a Bag-of-Words model.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# Sample data
X = ["Free money now", "Hello friend", "Win a lottery prize", "Meeting at 3pm"]
y = [1, 0, 1, 0]  # 1 = spam, 0 = ham

# Convert text data into Bag-of-Words vectors
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X).toarray()

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
```

---

#### Step 2: **Model Definition with `nn.Linear`**

We'll create a simple feedforward neural network with one hidden layer using `nn.Linear`. The number of input features will match the length of the BoW vectors.

```python
# Define the neural network model
class SpamDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SpamDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Output layer
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)  # Output layer (no activation for logits)
        return x

input_size = X_train.shape[1]  # Number of input features (from BoW)
hidden_size = 64  # Hyperparameter
num_classes = 2  # Spam or Ham

# Initialize the model
model = SpamDetector(input_size, hidden_size, num_classes)
```

---

#### Step 3: **Loss Function and Optimizer**

We'll use `nn.CrossEntropyLoss` as the loss function since this is a classification problem, and the predictions are logits (raw outputs).

```python
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
```

---

#### Step 4: **Training the Model**

The training process involves multiple epochs where the model learns by minimizing the loss using backpropagation and gradient descent.

```python
# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the weights
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

---

#### Step 5: **Evaluation**

After training, you can evaluate the model on the test data by predicting the labels and comparing them with the true labels.

```python
# Evaluate the model
model.eval()  # Set the model to evaluation mode

with torch.no_grad():  # No need to calculate gradients during testing
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)  # Get predicted class
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    
print(f'Accuracy: {accuracy * 100:.2f}%')
```

---

### 3. **Understanding the Components**:

#### a. **`nn.Linear`**:
- **Purpose**: This is a fully connected layer that applies a linear transformation to the input.
  - `nn.Linear(input_features, output_features)` where `input_features` is the number of features in the input data and `output_features` is the number of neurons in the next layer.

#### b. **`nn.CrossEntropyLoss`**:
- **Purpose**: This loss function combines `nn.LogSoftmax` and `nn.NLLLoss` in one single class, ideal for multi-class classification tasks.
  - It expects the raw logits as input and internally applies the softmax function to convert them into probabilities.

#### c. **Activation Function (`ReLU`)**:
- **Purpose**: The Rectified Linear Unit (ReLU) activation function introduces non-linearity into the model, which helps in learning complex patterns.

---

### 4. **Conclusion**
This basic spam detection model demonstrates how to use PyTorch's `nn.Linear` for building a simple feedforward network and `nn.CrossEntropyLoss` for classification tasks. You can experiment by adjusting the architecture, using different vectorization methods like TF-IDF or embeddings, or applying more complex deep learning models (e.g., LSTMs).