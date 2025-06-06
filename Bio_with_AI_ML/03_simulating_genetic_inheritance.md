To implement a simple neural network that simulates genetic crosses based on Mendelian genetics, we can create a model that predicts the probability of inheriting certain traits based on the genotypes of the parents. For simplicity, let's consider a single trait with two alleles (e.g., A and a) and two parents that can have the following genotypes:

- AA (homozygous dominant)
- Aa (heterozygous)
- aa (homozygous recessive)

We'll create a neural network that takes the genotypes of the two parents as input and predicts the probabilities of the offspring having specific genotypes (AA, Aa, and aa).

### Step-by-Step Implementation

#### 1. **Setting Up the Environment**

Make sure you have the necessary libraries installed:

```bash
pip install torch torchvision numpy
```

#### 2. **Define the Dataset**

We'll create a dataset representing the combinations of parent genotypes and the corresponding probabilities of the offspring genotypes.

```python
import torch
import numpy as np
import pandas as pd

# Create a dataset
data = {
    'Parent1': ['AA', 'AA', 'Aa', 'Aa', 'aa', 'aa'],
    'Parent2': ['AA', 'Aa', 'AA', 'aa', 'Aa', 'aa'],
    'AA': [1.0, 0.5, 0.5, 0.0, 0.0, 0.0],
    'Aa': [0.0, 0.5, 0.5, 0.5, 0.5, 0.0],
    'aa': [0.0, 0.0, 0.0, 0.5, 0.5, 1.0]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode genotypes to numerical values
def encode_genotype(genotype):
    if genotype == 'AA':
        return [1, 0, 0]  # AA
    elif genotype == 'Aa':
        return [0, 1, 0]  # Aa
    elif genotype == 'aa':
        return [0, 0, 1]  # aa

# Prepare features and labels
X = []
y = []

for index, row in df.iterrows():
    parent1_encoded = encode_genotype(row['Parent1'])
    parent2_encoded = encode_genotype(row['Parent2'])
    X.append(parent1_encoded + parent2_encoded)  # Combine parent genotypes
    y.append([row['AA'], row['Aa'], row['aa']])  # Probabilities

X = np.array(X)
y = np.array(y)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)
```

#### 3. **Define the Neural Network Model**

We'll create a simple feedforward neural network.

```python
import torch.nn as nn
import torch.optim as optim

class GeneticsNN(nn.Module):
    def __init__(self):
        super(GeneticsNN, self).__init__()
        self.fc1 = nn.Linear(4, 8)  # Input layer
        self.fc2 = nn.Linear(8, 3)   # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Hidden layer with ReLU activation
        x = torch.softmax(self.fc2(x), dim=1)  # Output layer with softmax activation
        return x

# Instantiate the model
model = GeneticsNN()
```

#### 4. **Train the Model**

Set up the loss function and optimizer, and then train the model.

```python
# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients

    outputs = model(X_tensor)  # Forward pass
    loss = criterion(outputs, y_tensor)  # Calculate loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:  # Print loss every 100 epochs
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
```

#### 5. **Evaluate the Model**

After training, you can evaluate the model's predictions based on test inputs.

```python
# Function to predict probabilities of offspring genotypes
def predict_probabilities(parent1, parent2):
    encoded_parent1 = encode_genotype(parent1)
    encoded_parent2 = encode_genotype(parent2)
    input_tensor = torch.FloatTensor(encoded_parent1 + encoded_parent2).unsqueeze(0)  # Add batch dimension
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        probabilities = model(input_tensor).numpy()
    return probabilities

# Example predictions
print("Predicted Probabilities:")
print(predict_probabilities('Aa', 'Aa'))  # Example: Aa x Aa
print(predict_probabilities('AA', 'aa'))   # Example: AA x aa
print(predict_probabilities('aa', 'aa'))   # Example: aa x aa
```

### Conclusion

This implementation demonstrates how to create a simple neural network to predict the probabilities of inheriting certain traits based on the genotypes of the parents using PyTorch. You can experiment with more complex models, additional traits, or different training techniques to enhance the accuracy of predictions. 

If you have any further questions or need additional features, feel free to ask!