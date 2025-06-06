To automate hyperparameter tuning for a PyTorch model using Bash scripts, you can run experiments with different hyperparameter values by iterating through combinations of values in a Bash script. Each combination will train the model with a different set of hyperparameters, allowing you to find the optimal configuration.

Here’s how you can set up a Bash script to automate hyperparameter tuning for a PyTorch model:

### Approach

1. **Define Hyperparameter Ranges**: Set ranges or lists of values for the hyperparameters you want to tune.
2. **Bash Script for Running Experiments**: Use loops in Bash to try different combinations of hyperparameters.
3. **Python Script to Train the Model**: This script accepts hyperparameters as arguments, trains the model, and logs performance.

### Step 1: Python Script to Train the Model (`train_model.py`)

This Python script:
- Loads data and initializes the model.
- Accepts hyperparameters as command-line arguments.
- Trains the model, evaluates it, and outputs performance metrics.

```python
import argparse
import torch
import torch.optim as optim
from model import MyModel  # Replace with your model import
from dataset import get_dataloader  # Replace with your dataset import

# Argument parser for hyperparameters
parser = argparse.ArgumentParser(description="PyTorch Model Training with Hyperparameter Tuning")
parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for training")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")
parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay (L2 regularization)")

args = parser.parse_args()

# Initialize model, loss, and optimizer
model = MyModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# Load data
train_loader, val_loader = get_dataloader(batch_size=args.batch_size)

# Training loop
for epoch in range(args.num_epochs):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
accuracy = 0
with torch.no_grad():
    for data, target in val_loader:
        output = model(data)
        predictions = output.argmax(dim=1)
        accuracy += (predictions == target).sum().item()

accuracy = accuracy / len(val_loader.dataset)
print(f"Accuracy: {accuracy}")
```

### Step 2: Bash Script for Automating Hyperparameter Tuning (`tune_hyperparameters.sh`)

This Bash script will:
1. Define hyperparameter values to try.
2. Loop through different combinations of hyperparameters.
3. Run the training Python script for each combination, logging the results.

```bash
#!/bin/bash

# Define hyperparameter ranges
LEARNING_RATES=("0.01" "0.001" "0.0001")
BATCH_SIZES=("32" "64" "128")
EPOCHS=("10" "20")
WEIGHT_DECAYS=("0" "0.0001" "0.001")

# Log file for recording results
LOG_FILE="tuning_results.txt"
echo "Hyperparameter Tuning Results" > $LOG_FILE

# Loop over each hyperparameter combination
for LR in "${LEARNING_RATES[@]}"; do
  for BS in "${BATCH_SIZES[@]}"; do
    for EPOCH in "${EPOCHS[@]}"; do
      for WD in "${WEIGHT_DECAYS[@]}"; do
        echo "Running training with LR=$LR, Batch Size=$BS, Epochs=$EPOCH, Weight Decay=$WD"

        # Run the training script with the current hyperparameters
        RESULT=$(python3 train_model.py --learning_rate "$LR" --batch_size "$BS" --num_epochs "$EPOCH" --weight_decay "$WD")

        # Log the hyperparameters and result
        echo "LR=$LR, Batch Size=$BS, Epochs=$EPOCH, Weight Decay=$WD, $RESULT" >> $LOG_FILE
      done
    done
  done
done

echo "Hyperparameter tuning completed. Results saved in $LOG_FILE"
```

### Explanation of the Bash Script

1. **Define Hyperparameter Values**:
   - Lists of possible values for learning rate (`LEARNING_RATES`), batch size (`BATCH_SIZES`), epochs (`EPOCHS`), and weight decay (`WEIGHT_DECAYS`).
  
2. **Loop Through Hyperparameter Combinations**:
   - Nested `for` loops iterate through all combinations of hyperparameters.
  
3. **Run Training Script and Capture Results**:
   - Each iteration calls `train_model.py` with the current set of hyperparameters.
   - The `RESULT` variable stores the output (accuracy in this case), which is logged to a `tuning_results.txt` file.

### Running the Script

1. Save the Bash script as `tune_hyperparameters.sh`.
2. Make it executable:

   ```bash
   chmod +x tune_hyperparameters.sh
   ```

3. Run the script:

   ```bash
   ./tune_hyperparameters.sh
   ```

This script will train the model with different hyperparameter configurations, logging the results and hyperparameters for each run. This approach lets you review results in `tuning_results.txt` to identify the best-performing configuration.