# Debugging PyTorch Models

## Overview
Debugging and visualizing PyTorch models is essential for understanding model behavior, optimizing performance, and identifying potential issues. Tools like `torchviz` and `TensorBoard` are invaluable for visualizing computational graphs and monitoring training progress. This document outlines how to use these tools effectively.

## 1. **Using `torchviz`**

`torchviz` is a Python library that helps visualize the computational graph of PyTorch models. It generates a graph representation of the model's forward pass, allowing you to inspect the flow of tensors and operations.

### 1.1 Installation
```bash
pip install torchviz
```

### 1.2 Basic Usage
To visualize the computational graph using `torchviz`, you need to create a forward pass through your model with some input data.

#### Example
```python
import torch
import torch.nn as nn
from torchviz import make_dot

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Instantiate the model and create a sample input
model = SimpleModel()
input_tensor = torch.randn(1, 10)  # Batch size of 1, 10 features

# Perform a forward pass
output = model(input_tensor)

# Generate and visualize the computational graph
dot = make_dot(output, params=dict(list(model.named_parameters())))
dot.render("model_graph", format="png")  # Save as PNG
```

The resulting graph file `model_graph.png` will show the model's structure, including layers and operations.

## 2. **Using TensorBoard**

TensorBoard is a powerful visualization tool that provides insights into model training, including metrics, graphs, and images. PyTorch supports TensorBoard through the `torch.utils.tensorboard` module.

### 2.1 Installation
```bash
pip install tensorboard
```

### 2.2 Basic Usage
You can log various metrics, such as loss and accuracy, and visualize the model graph in TensorBoard.

#### Example
```python
from torch.utils.tensorboard import SummaryWriter

# Create a SummaryWriter
writer = SummaryWriter('runs/experiment_1')

# Log model graph
writer.add_graph(model, input_tensor)

# Training loop example
for epoch in range(10):  # Simulating training for 10 epochs
    # Simulate loss value
    loss = torch.randn(1).item()  # Random loss value for demonstration
    writer.add_scalar('Loss/train', loss, epoch)

writer.close()
```

### 2.3 Viewing TensorBoard
After logging your metrics and model graph, you can view them using TensorBoard.

```bash
tensorboard --logdir=runs
```

Open your browser and navigate to `http://localhost:6006` to see the TensorBoard interface.

## 3. **Debugging Tips**

- **Check Gradients**: Ensure that gradients are flowing correctly through the model. Use `torch.autograd.gradcheck` for numerical gradient checking.
- **Print Shapes**: Print the shapes of tensors at different points in the model to catch shape mismatches.
- **Set Breakpoints**: Use Python debuggers like `pdb` to set breakpoints and inspect the model during the forward and backward passes.
- **Use Hooks**: Attach hooks to layers to inspect inputs, outputs, and gradients.
  
#### Example of Using Hooks
```python
def hook_fn(module, input, output):
    print(f"Layer: {module.__class__.__name__}, Input Shape: {input[0].shape}, Output Shape: {output.shape}")

# Register a hook on the first layer
hook = model.fc1.register_forward_hook(hook_fn)

# Forward pass
output = model(input_tensor)

# Remove the hook
hook.remove()
```

## Conclusion
Debugging and visualizing PyTorch models using `torchviz` and TensorBoard can greatly enhance your understanding of model behavior and performance. By incorporating these tools into your workflow, you can more effectively diagnose issues, optimize performance, and track model training progress.
