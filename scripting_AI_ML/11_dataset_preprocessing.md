To preprocess datasets like MNIST or CIFAR-10 for PyTorch model training, you can create a Bash script that automates the downloading, extraction, and transformation of the datasets into a format suitable for training. This script can use Python to handle the actual preprocessing tasks, such as normalization and data augmentation.

### Step 1: Create a Python Script for Preprocessing

First, create a Python script that will handle the preprocessing of the dataset. This script will download the dataset, apply necessary transformations, and save it for use in training.

**`preprocess_data.py`**:

```python
import os
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def preprocess_mnist(data_dir):
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize for MNIST
    ])
    
    # Download and load the training data
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def preprocess_cifar10(data_dir):
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for CIFAR-10
    ])
    
    # Download and load the training data
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1]
    dataset_type = sys.argv[2].lower()
    
    if dataset_type == 'mnist':
        train_loader, test_loader = preprocess_mnist(data_dir)
        print("MNIST dataset preprocessed successfully.")
    elif dataset_type == 'cifar10':
        train_loader, test_loader = preprocess_cifar10(data_dir)
        print("CIFAR-10 dataset preprocessed successfully.")
    else:
        print("Invalid dataset type. Please choose 'mnist' or 'cifar10'.")
```

### Step 2: Create the Bash Script for Preprocessing

Next, create a Bash script that will call the Python script, specifying the dataset type and data directory.

**`preprocess_data.sh`**:

```bash
#!/bin/bash

# Set the directory where datasets will be stored
DATA_DIR="./data"

# Create the data directory if it doesn't exist
if [[ ! -d "$DATA_DIR" ]]; then
    mkdir -p "$DATA_DIR"
fi

# Check command-line arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_type>"
    echo "<dataset_type> should be 'mnist' or 'cifar10'"
    exit 1
fi

DATASET_TYPE=$1

# Run the Python preprocessing script
python preprocess_data.py "$DATA_DIR" "$DATASET_TYPE"

# Check if preprocessing was successful
if [[ $? -eq 0 ]]; then
    echo "Dataset preprocessing completed successfully!"
else
    echo "Error during dataset preprocessing!"
    exit 1
fi
```

### Step 3: Make the Bash Script Executable

Make the Bash script executable by running the following command:

```bash
chmod +x preprocess_data.sh
```

### Step 4: Run the Bash Script

Now you can run the Bash script to preprocess the desired dataset. For example, to preprocess the MNIST dataset, you would execute:

```bash
./preprocess_data.sh mnist
```

To preprocess the CIFAR-10 dataset, you would execute:

```bash
./preprocess_data.sh cifar10
```

### Explanation of Key Components

1. **Python Script (`preprocess_data.py`)**:
   - Defines functions to preprocess MNIST and CIFAR-10 datasets, including transformations and data normalization.
   - Downloads the dataset if it isn’t already available in the specified directory.
   - Returns DataLoaders for training and test datasets.

2. **Bash Script (`preprocess_data.sh`)**:
   - Checks for the existence of a directory to store the dataset.
   - Validates command-line arguments to specify the dataset type.
   - Calls the Python preprocessing script with the data directory and dataset type as arguments.
   - Outputs success or error messages based on the completion of the preprocessing step.

### Additional Considerations

- **Dependencies**: Ensure that you have the required libraries installed, such as `torch` and `torchvision`. You can install them via pip if needed:

  ```bash
  pip install torch torchvision
  ```

- **Batch Size and Normalization**: Adjust the batch size and normalization values based on your model requirements and dataset characteristics. 

This setup provides an automated way to preprocess datasets like MNIST and CIFAR-10 for use in training PyTorch models, simplifying the preparation process.