To automate training PyTorch models on GPU with a fallback to CPU if a GPU is unavailable, you can use a Python script that detects the availability of a GPU. This script will attempt to train on the GPU if available, or otherwise fall back to CPU.

Here's a script to accomplish this:

### Step 1: Training Script with GPU Fallback (`train_with_gpu_fallback.py`)

This Python script:
- Checks if a GPU is available.
- Loads the model and data.
- Trains the model on the available device.

```python
import torch
import argparse
from model import MyModel  # Replace with your model
from dataset import get_dataloader  # Replace with your data loading function

# Argument parser for hyperparameters
parser = argparse.ArgumentParser(description="Train PyTorch Model with GPU Fallback")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay (L2 regularization)")

args = parser.parse_args()

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# Initialize model, loss function, and optimizer
model = MyModel().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# Load data
train_loader, val_loader = get_dataloader(batch_size=args.batch_size)

# Training loop
for epoch in range(args.num_epochs):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{args.num_epochs} completed")

# Evaluate the model
model.eval()
accuracy = 0
with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        predictions = output.argmax(dim=1)
        accuracy += (predictions == target).sum().item()

accuracy = accuracy / len(val_loader.dataset)
print(f"Model Accuracy: {accuracy}")
```

### Step 2: Bash Script to Automate the Training Script (`train_automate.sh`)

This Bash script:
- Calls the training script and redirects logs to a file.
- Prints a summary upon completion.

```bash
#!/bin/bash

# Training hyperparameters
LEARNING_RATE=0.001
BATCH_SIZE=64
EPOCHS=10
WEIGHT_DECAY=0.0001

# Run the training script and log output
python3 train_with_gpu_fallback.py --learning_rate "$LEARNING_RATE" --batch_size "$BATCH_SIZE" --num_epochs "$EPOCHS" --weight_decay "$WEIGHT_DECAY" | tee training_log.txt

echo "Training completed. Logs saved in training_log.txt"
```

### Explanation of the Components

1. **Device Detection**: The `torch.device("cuda" if torch.cuda.is_available() else "cpu")` line in the Python script sets the device to GPU if available, otherwise defaults to CPU.
2. **Data and Model Transfer**: `.to(device)` is used to send data and the model to the correct device (GPU or CPU).
3. **Logging**: The Bash script logs the output to `training_log.txt` for later review.
4. **Hyperparameters**: Adjustable in the Bash script, enabling easy tuning without modifying the Python code.

### Running the Script

1. Save the Python script as `train_with_gpu_fallback.py`.
2. Save the Bash script as `train_automate.sh` and make it executable:

   ```bash
   chmod +x train_automate.sh
   ```

3. Run the script:

   ```bash
   ./train_automate.sh
   ```

This setup will automatically attempt to train on GPU if available, falling back to CPU if not, and log the process in `training_log.txt`.