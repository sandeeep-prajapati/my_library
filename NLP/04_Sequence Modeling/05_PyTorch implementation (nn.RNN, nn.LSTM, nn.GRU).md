Here's an overview of the implementations for Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU) using PyTorch. We will use a simple text classification task as an example (e.g., spam detection). The implementation will include data preprocessing, model definition, training, and evaluation.

### 1. **Data Preprocessing**

First, we need to preprocess our data. For simplicity, we will create a small sample dataset and use Bag-of-Words (BoW) for encoding the text.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Sample data
X = ["Free money now", "Hello friend", "Win a lottery prize", "Meeting at 3pm"]
y = [1, 0, 1, 0]  # 1 = spam, 0 = ham

# Convert text data into Bag-of-Words vectors
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X).toarray()

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
```

### 2. **Model Definitions**

We will define three models: RNN, LSTM, and GRU.

#### RNN Model

```python
class RNNSpamDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNSpamDetector, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x.unsqueeze(1), h0)  # Add batch dimension
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out
```

#### LSTM Model

```python
class LSTMSpamDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMSpamDetector, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))  # Add batch dimension
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out
```

#### GRU Model

```python
class GRUSpamDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUSpamDetector, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x.unsqueeze(1), h0)  # Add batch dimension
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out
```

### 3. **Training and Evaluation**

We'll define a common training and evaluation process for all three models.

#### Training Function

```python
def train_model(model, X_train, y_train, num_epochs=100, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
    return accuracy
```

### 4. **Running the Models**

Now, we can instantiate and train each model, then evaluate their performance.

#### Train and Evaluate RNN

```python
# Define model parameters
input_size = X_train.shape[1]
hidden_size = 64
num_layers = 1
num_classes = 2

# RNN
rnn_model = RNNSpamDetector(input_size, hidden_size, num_layers, num_classes)
print("Training RNN...")
train_model(rnn_model, X_train, y_train)
rnn_accuracy = evaluate_model(rnn_model, X_test, y_test)
print(f'RNN Accuracy: {rnn_accuracy * 100:.2f}%')
```

#### Train and Evaluate LSTM

```python
# LSTM
lstm_model = LSTMSpamDetector(input_size, hidden_size, num_layers, num_classes)
print("\nTraining LSTM...")
train_model(lstm_model, X_train, y_train)
lstm_accuracy = evaluate_model(lstm_model, X_test, y_test)
print(f'LSTM Accuracy: {lstm_accuracy * 100:.2f}%')
```

#### Train and Evaluate GRU

```python
# GRU
gru_model = GRUSpamDetector(input_size, hidden_size, num_layers, num_classes)
print("\nTraining GRU...")
train_model(gru_model, X_train, y_train)
gru_accuracy = evaluate_model(gru_model, X_test, y_test)
print(f'GRU Accuracy: {gru_accuracy * 100:.2f}%')
```

### Summary

1. **RNN**: Basic recurrent neural network; often suffers from vanishing gradients.
2. **LSTM**: More complex architecture that can learn long-term dependencies, handling vanishing gradient issues effectively.
3. **GRU**: Similar to LSTM but with fewer parameters and simpler architecture.

This implementation demonstrates how to build, train, and evaluate RNN, LSTM, and GRU models in PyTorch for a text classification task. You can further expand and customize the models, preprocessing steps, and datasets as needed for your specific application.