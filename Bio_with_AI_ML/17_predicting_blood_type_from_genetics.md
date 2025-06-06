To predict blood types based on genetic markers from DNA sequences, you can build a simple supervised classification model that learns relationships between specific alleles and blood group phenotypes (A, B, AB, O). Blood types are primarily determined by variations in the *ABO* gene, which has different alleles for the A, B, and O blood types.

### Step-by-Step Guide to Building the Classifier

1. **Data Collection**:
   - Collect a dataset of DNA sequences with known blood types, focusing on the *ABO* gene's relevant alleles.
   - For a simple classification model, represent each sequence as a feature vector based on alleles associated with each blood type (e.g., presence of specific SNPs or nucleotide patterns in *ABO* alleles).

2. **Data Preprocessing**:
   - Convert DNA sequences into a format suitable for machine learning by encoding alleles or specific genetic markers (SNPs) as categorical variables or binary features.
   - Encode the blood types as numeric labels (e.g., `0` for O, `1` for A, `2` for B, `3` for AB).

3. **Model Architecture**:
   - A basic fully connected neural network or logistic regression model should be effective given that the features will be limited to a few genetic markers.
   - Since there are only a few blood types, a simple softmax output layer can handle the multi-class classification task.

4. **Implementation in PyTorch**:

Below is the PyTorch code to implement a simple fully connected neural network for predicting blood types based on genetic markers.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Example dataset of genetic markers and blood types
# Replace this with actual genetic marker data for the ABO gene
data = pd.DataFrame({
    "allele_A": [1, 0, 1, 0, 1],
    "allele_B": [0, 1, 0, 1, 0],
    "allele_O": [0, 0, 1, 1, 0],
    "blood_type": [1, 2, 0, 3, 1]  # Blood types as labels (O=0, A=1, B=2, AB=3)
})

# Separate features and target variable
X = data[['allele_A', 'allele_B', 'allele_O']].values
y = data['blood_type'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

# Define the neural network model
class BloodTypeClassifier(nn.Module):
    def __init__(self):
        super(BloodTypeClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 8)
        self.fc2 = nn.Linear(8, 4)  # 4 classes for blood types

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = BloodTypeClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate model accuracy on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = accuracy_score(y_test, predicted)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

### Explanation of Code

1. **Data Loading and Preprocessing**:
   - The example dataset has three features corresponding to genetic markers for each allele (A, B, O).
   - Blood type is encoded as a numerical class (e.g., `0` for O, `1` for A, `2` for B, and `3` for AB).

2. **Model Architecture**:
   - The model has a single hidden layer with `ReLU` activation and an output layer with four neurons for the four blood type classes.

3. **Training Loop**:
   - The model trains over 100 epochs using `Adam` as the optimizer and `CrossEntropyLoss` as the criterion for multi-class classification.
   - Training loss is printed every 10 epochs to monitor progress.

4. **Evaluation**:
   - The model’s accuracy is calculated on the test set by comparing predictions with true labels.

### Next Steps

- **Data Augmentation**: Increase the dataset size if possible, as more examples of genetic markers for each blood type will improve model generalization.
- **Cross-Validation**: Use k-fold cross-validation to improve model robustness.
- **Hyperparameter Tuning**: Experiment with different learning rates, hidden layer sizes, and batch sizes to optimize model performance.

This model provides a basic framework for predicting blood type based on genetic markers. With a more extensive dataset of genetic markers and labels, this approach could be extended to real-world applications in genetics and personalized medicine.