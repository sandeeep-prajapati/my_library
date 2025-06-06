Automating the logging of training metrics, such as accuracy and loss values, in PyTorch using Bash scripts involves creating a Python training script that handles the logging, along with a Bash script that executes the training process. Below is a step-by-step guide to accomplish this.

### Step 1: Create a Python Script for Training and Logging

First, create a Python script that will perform the training of your model and log the metrics to a file.

**`train.py`**:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import logging
import os
import sys

# Set up logging
def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s %(levelname)s:%(message)s')

def train_model(epochs, log_file):
    # Setup logging
    setup_logging(log_file)

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Define model, loss function, and optimizer
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        # Log metrics
        logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <epochs> <log_file>")
        sys.exit(1)

    epochs = int(sys.argv[1])
    log_file = sys.argv[2]
    
    train_model(epochs, log_file)
```

### Step 2: Create the Bash Script to Automate Training

Next, create a Bash script that runs the Python training script, passing the number of epochs and log file name as arguments.

**`run_training.sh`**:

```bash
#!/bin/bash

# Check command-line arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <epochs> <log_file>"
    exit 1
fi

EPOCHS=$1
LOG_FILE=$2

# Run the training script
python train.py "$EPOCHS" "$LOG_FILE"

# Check if training was successful
if [[ $? -eq 0 ]]; then
    echo "Training completed successfully! Metrics logged to $LOG_FILE."
else
    echo "Error during training!"
    exit 1
fi
```

### Step 3: Make the Bash Script Executable

Make the Bash script executable by running the following command:

```bash
chmod +x run_training.sh
```

### Step 4: Run the Bash Script

Now you can run the Bash script to start the training process and log the metrics. For example:

```bash
./run_training.sh 10 training_metrics.log
```

### Explanation of Key Components

1. **Python Script (`train.py`)**:
   - Sets up logging to a specified file.
   - Loads the MNIST dataset and defines a simple neural network model.
   - Trains the model for a specified number of epochs while calculating and logging the loss and accuracy after each epoch.

2. **Bash Script (`run_training.sh`)**:
   - Validates command-line arguments to ensure the correct number of parameters are provided.
   - Calls the Python script with the provided arguments for epochs and log file name.
   - Checks if the training was successful and outputs the result.

### Additional Considerations

- **Logging Format**: You can customize the logging format in the `setup_logging` function based on your needs.

- **Model Complexity**: The provided model is a simple feedforward neural network for demonstration. You can replace it with a more complex model as required.

- **Dataset**: The script uses the MNIST dataset for simplicity. Modify the dataset loading and transformations to suit your application.

- **Dependencies**: Ensure you have the necessary libraries installed. You can install them using:

  ```bash
  pip install torch torchvision
  ```

This setup allows you to automate the training of a PyTorch model and log the training metrics, such as loss and accuracy, efficiently.