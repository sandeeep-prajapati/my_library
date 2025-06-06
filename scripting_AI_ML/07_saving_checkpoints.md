To automate saving model checkpoints during long training sessions in PyTorch, you can modify your Python training script to save checkpoints periodically and use a Bash script to manage these training sessions. This will help you resume training in case of interruptions and monitor progress.

Here’s how to set up checkpointing in your PyTorch training script, and a Bash script to execute it.

### Step 1: Modify Python Script to Save Checkpoints (`train_with_checkpoints.py`)

This script will:
- Save model checkpoints at specified intervals (e.g., every `n` epochs).
- Save model weights, optimizer state, and the current epoch to enable resuming training.

```python
import torch
import argparse
import os
from model import MyModel  # Replace with your model
from dataset import get_dataloader  # Replace with your data loading function

# Argument parser for hyperparameters
parser = argparse.ArgumentParser(description="Train PyTorch Model with Checkpointing")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay (L2 regularization)")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every n epochs")

args = parser.parse_args()

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# Initialize model, loss, and optimizer
model = MyModel().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# Load data
train_loader, val_loader = get_dataloader(batch_size=args.batch_size)

# Create checkpoint directory if it doesn't exist
os.makedirs(args.checkpoint_dir, exist_ok=True)

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

    print(f"Epoch {epoch+1}/{args.num_epochs} completed with loss: {loss.item()}")

    # Save checkpoint at specified intervals
    if (epoch + 1) % args.save_interval == 0:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

print("Training completed.")
```

### Step 2: Bash Script to Automate Training with Checkpoints (`train_with_checkpoints.sh`)

This Bash script:
- Runs the training script and monitors its output.
- Logs all training information, including checkpoint saves, into a file.

```bash
#!/bin/bash

# Training hyperparameters
LEARNING_RATE=0.001
BATCH_SIZE=64
EPOCHS=50
WEIGHT_DECAY=0.0001
CHECKPOINT_DIR="checkpoints"
SAVE_INTERVAL=5

# Create directory for checkpoints if not exists
mkdir -p "$CHECKPOINT_DIR"

# Run the training script and log the output
python3 train_with_checkpoints.py --learning_rate "$LEARNING_RATE" \
                                  --batch_size "$BATCH_SIZE" \
                                  --num_epochs "$EPOCHS" \
                                  --weight_decay "$WEIGHT_DECAY" \
                                  --checkpoint_dir "$CHECKPOINT_DIR" \
                                  --save_interval "$SAVE_INTERVAL" | tee training_log.txt

echo "Training with checkpointing completed. Logs saved in training_log.txt"
```

### Explanation of the Components

1. **Checkpoint Directory**: `--checkpoint_dir` specifies where to save checkpoints, and the directory is created automatically if it doesn’t exist.
2. **Checkpoint Saving in the Python Script**:
   - Every `n` epochs (set by `--save_interval`), a checkpoint is saved, containing the model state, optimizer state, and current epoch.
   - This enables you to resume training from the last checkpoint.
3. **Logging**: The Bash script logs all training outputs, including loss values and checkpoint saves, to `training_log.txt`.

### Running the Script

1. Save the Python script as `train_with_checkpoints.py`.
2. Save the Bash script as `train_with_checkpoints.sh` and make it executable:

   ```bash
   chmod +x train_with_checkpoints.sh
   ```

3. Run the script:

   ```bash
   ./train_with_checkpoints.sh
   ```

### Resuming Training from a Checkpoint (Optional)

To resume training, you can modify the `train_with_checkpoints.py` script to load the latest checkpoint:

```python
# Resume training if a checkpoint exists
if os.path.exists(latest_checkpoint_path):
    checkpoint = torch.load(latest_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0
```

This setup allows you to monitor training progress, save checkpoints at intervals, and resume from the last saved checkpoint if needed.