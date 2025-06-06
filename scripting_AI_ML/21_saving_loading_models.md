Automating the saving and loading of PyTorch models using a Bash script can streamline the training process, allowing you to resume training from a checkpoint when needed. Below is a comprehensive example that includes a Bash script for managing the training process, along with a Python script that handles model training, saving, and loading.

### Bash Script for Automating Model Saving and Loading

This script will:
1. Set up the environment.
2. Install necessary packages.
3. Start training and save the model checkpoints periodically.

**`train_model.sh`**:

```bash
#!/bin/bash

# Configuration variables
MODEL_PATH="./models/my_model.pth"
LOG_DIR="./logs"
EPOCHS=10
LEARNING_RATE=0.001
CHECKPOINT_INTERVAL=5  # Save checkpoint every 5 epochs

# Create directories if they do not exist
mkdir -p $LOG_DIR
mkdir -p ./models

# Install required packages
if ! python -c "import torch" &> /dev/null; then
    echo "Installing required packages..."
    pip install torch torchvision
fi

# Check if a model checkpoint exists
if [ -f $MODEL_PATH ]; then
    echo "Loading existing model from $MODEL_PATH"
    RESUME="--resume $MODEL_PATH"
else
    echo "No existing model found. Starting fresh training."
    RESUME=""
fi

# Start training
echo "Starting training..."
python train.py --epochs $EPOCHS --learning_rate $LEARNING_RATE --log_dir $LOG_DIR $RESUME

# Check if the training was successful
if [ $? -ne 0 ]; then
    echo "Error during training"
    exit 1
fi

echo "Training completed successfully!"
```

### Python Script for Training with Checkpoint Saving and Loading

This Python script handles the model training, including saving and loading logic.

**`train.py`**:

```python
import argparse
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

def train_model(epochs, learning_rate, log_dir, resume=None):
    # Define data transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # Load dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Load model
    model = models.resnet18(num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load from checkpoint if provided
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    # Training loop
    model.train()
    for epoch in range(start_epoch, epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:  # Log every 100 batches
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Save checkpoint at specified intervals
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = f"./models/my_model_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Final save
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, './models/my_model.pth')
    print("Final model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PyTorch model with saving and loading functionality.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for logs')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training')

    args = parser.parse_args()

    train_model(args.epochs, args.learning_rate, args.log_dir, args.resume)
```

### Step 1: Make the Bash Script Executable

Make the Bash script executable by running:

```bash
chmod +x train_model.sh
```

### Step 2: Run the Bash Script

Execute the script to start the training process, which will automatically handle saving and loading the model:

```bash
./train_model.sh
```

### Explanation of Key Components

1. **Directory Setup**:
   - The script creates necessary directories for logs and models if they don't exist.

2. **Package Installation**:
   - It checks for the required Python packages (PyTorch) and installs them if they are not found.

3. **Model Loading**:
   - The script checks if a model checkpoint exists. If found, it resumes training from that checkpoint; otherwise, it starts fresh.

4. **Training Logic**:
   - The training script implements the logic for training a PyTorch model. It logs training metrics and saves checkpoints at specified intervals.

5. **Checkpoint Saving**:
   - After every specified number of epochs (`CHECKPOINT_INTERVAL`), the model's state and optimizer's state are saved to a checkpoint file. A final model save occurs after training completes.

6. **Error Handling**:
   - The script checks if the training process completed successfully and provides error messages if any issues arise.

### Additional Considerations

- **Checkpoint Management**: You may want to implement a mechanism to manage old checkpoints, such as deleting them or keeping only the last few.
- **Logging**: Consider adding logging to a file for easier monitoring.
- **Hyperparameter Customization**: You can modify the script to accept hyperparameters from the command line or a configuration file.

This setup enables efficient automation for saving and loading PyTorch models, facilitating easier training resumption and management of model states during training sessions.