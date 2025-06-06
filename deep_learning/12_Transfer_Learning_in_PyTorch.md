# Transfer Learning in PyTorch

## Overview
Transfer learning leverages a pretrained model on a large dataset to solve a different but related problem with less data and computing resources. Instead of training a model from scratch, we fine-tune a pretrained model (e.g., ResNet, VGG) by either modifying its final layers or adjusting its weights to adapt to a new task.

This section will focus on using transfer learning in PyTorch, covering how to load pretrained models, modify them for specific tasks, and fine-tune the weights.

## Steps for Transfer Learning

### 1. **Loading a Pretrained Model**
   PyTorch provides several pretrained models, such as ResNet, VGG, and Inception, through the `torchvision.models` module. These models are trained on the ImageNet dataset and can be fine-tuned for various tasks.

   ```python
   import torch
   import torchvision.models as models

   # Load a pretrained ResNet model
   model = models.resnet18(pretrained=True)
   ```

   **Common Pretrained Models**:
   - **ResNet**: Ideal for image classification tasks.
   - **VGG**: Known for its simplicity and deep architecture.
   - **Inception**: Well-suited for image classification with its efficient convolutional operations.

### 2. **Freezing the Pretrained Layers**
   In transfer learning, we can "freeze" the weights of the earlier layers of the model to preserve the learned features from the original task. This means that only the final layers, specific to the new task, will be trained.

   ```python
   # Freeze all layers except the last one
   for param in model.parameters():
       param.requires_grad = False
   ```

### 3. **Modifying the Final Layer**
   The final layer of the pretrained model is usually specific to the original dataset (e.g., 1000 output classes for ImageNet). To adapt the model to a new task, we replace this layer with one that matches the number of output classes for the new dataset.

   For example, if we are working with a dataset with 10 classes:

   ```python
   import torch.nn as nn

   # Modify the final layer to match the number of classes
   num_classes = 10
   model.fc = nn.Linear(model.fc.in_features, num_classes)
   ```

   For other models like VGG or Inception, you may need to adjust the last layer (e.g., `model.classifier` for VGG or `model.fc` for ResNet).

### 4. **Training the Model**
   With the final layer replaced, you can now fine-tune the model on your specific dataset. Typically, only the parameters of the final layer are updated, but you can also fine-tune some of the earlier layers by unfreezing them.

   ```python
   # Define loss function and optimizer
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

   # Training loop
   for epoch in range(num_epochs):
       for inputs, labels in dataloader:
           optimizer.zero_grad()  # Reset the gradients
           outputs = model(inputs)  # Forward pass
           loss = criterion(outputs, labels)  # Compute loss
           loss.backward()  # Backward pass
           optimizer.step()  # Update weights
   ```

### 5. **Fine-tuning the Entire Model**
   In some cases, you may want to fine-tune the entire model instead of just the final layer. To do this, you need to unfreeze some or all layers and allow them to be trained.

   ```python
   # Unfreeze all layers
   for param in model.parameters():
       param.requires_grad = True

   # Redefine the optimizer to update all layers
   optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
   ```

   Fine-tuning the entire model is more computationally expensive and may require a lower learning rate.

## Example: Transfer Learning with ResNet on CIFAR-10

Here is a full example of applying transfer learning using ResNet18 on the CIFAR-10 dataset:

```python
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

# Load the CIFAR-10 dataset
transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Freeze all layers except the final layer
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
```

## Conclusion
Transfer learning allows us to leverage the power of large pretrained models and adapt them to new tasks with smaller datasets. By freezing early layers and fine-tuning the final layers, we can achieve efficient learning while preserving the general features learned from the original task. Models like ResNet and VGG are great starting points for many image classification tasks, making transfer learning a powerful tool in deep learning workflows.
