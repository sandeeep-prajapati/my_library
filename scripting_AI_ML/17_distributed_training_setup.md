Setting up a PyTorch distributed training environment across multiple nodes requires configuring several components, including networking, environment variables, and the training script. Below is a Bash script that automates the setup for a distributed training environment using PyTorch.

### Bash Script to Set Up Distributed Training

This script will:
1. Install necessary packages.
2. Set environment variables for distributed training.
3. Launch the training script across multiple nodes.

**`setup_distributed_training.sh`**:

```bash
#!/bin/bash

# Configuration variables
NODE_IPS=("192.168.1.1" "192.168.1.2")  # Replace with actual IPs of your nodes
MASTER_ADDR="${NODE_IPS[0]}"
MASTER_PORT=12345
WORLD_SIZE=${#NODE_IPS[@]}  # Total number of nodes
NUM_NODES=${#NODE_IPS[@]}
NODE_RANK=$1  # Pass the rank of the current node (0 for master, 1 for the second node, etc.)
DATA_DIR="./data"
SAVE_DIR="./models"
EPOCHS=5
LEARNING_RATE=0.001
MODEL_NAME="resnet18"

# Check if the rank is provided
if [ -z "$NODE_RANK" ]; then
    echo "Usage: $0 <node_rank>"
    exit 1
fi

# Install PyTorch if not already installed
if ! python -c "import torch" &> /dev/null; then
    echo "Installing PyTorch..."
    pip install torch torchvision
fi

# Export environment variables for distributed training
export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE
export NODE_RANK
export DATA_DIR
export SAVE_DIR

# Launch the training script on the current node
echo "Starting distributed training on node $NODE_RANK..."
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$NUM_NODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_model.py --epochs $EPOCHS --learning_rate $LEARNING_RATE --model_name $MODEL_NAME --data_dir $DATA_DIR --save_dir $SAVE_DIR

# Check if the training was successful
if [ $? -ne 0 ]; then
    echo "Error during distributed training on node $NODE_RANK"
    exit 1
fi

echo "Distributed training completed successfully on node $NODE_RANK!"
```

### Step 1: Create the Training Script

You also need a training script that can handle distributed training. Here's a modified version of the earlier training script to work in a distributed setting.

**`train_model.py`**:

```python
import argparse
import torch
import torch.distributed as dist
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_model(rank, epochs, learning_rate, model_name, data_dir, save_dir):
    setup(rank, world_size)

    # Define data transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # Load dataset
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, sampler=train_sampler)

    # Load model
    model = models.resnet18(num_classes=10)
    model = model.to(rank)  # Move model to the current device
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # Shuffle data at each epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)  # Move data to the current device
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        if rank == 0:  # Print loss only on the master node
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Save model only on the master node
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'{model_name}_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PyTorch model in a distributed manner.")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='resnet18', help='Model name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for storing data')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory for saving models')

    args = parser.parse_args()

    world_size = int(os.environ['WORLD_SIZE'])
    node_rank = int(os.environ['NODE_RANK'])

    train_model(node_rank, args.epochs, args.learning_rate, args.model_name, args.data_dir, args.save_dir)
```

### Step 2: Make the Bash Script Executable

Make the Bash script executable by running:

```bash
chmod +x setup_distributed_training.sh
```

### Step 3: Run the Bash Script

To start the distributed training, run the script on each node, specifying the rank for each node:

On the master node (rank 0):
```bash
./setup_distributed_training.sh 0
```

On the second node (rank 1):
```bash
./setup_distributed_training.sh 1
```

### Explanation of Key Components

1. **Node Configuration**:
   - The script defines an array of node IPs and uses the first one as the master address.
   - The `WORLD_SIZE` is calculated based on the number of nodes.

2. **Environment Variables**:
   - The script exports environment variables necessary for PyTorch's distributed training.

3. **Distributed Training Launch**:
   - The script uses `torch.distributed.launch` to start the training script on each node.
   - Each node uses its rank to identify itself in the distributed system.

4. **Training Script (`train_model.py`)**:
   - Initializes the process group for distributed training.
   - Utilizes a `DistributedSampler` for loading the dataset to ensure that each node gets a different subset of the data.
   - Saves the model only on the master node to avoid overwriting.

### Additional Considerations

- **Firewall and Networking**: Ensure that the nodes can communicate over the specified port (e.g., 12345). You may need to configure firewall rules.
- **GPU Support**: If using GPUs, modify the code to handle CUDA devices accordingly.
- **Error Handling**: Consider adding more robust error handling for different scenarios (e.g., network failures, resource unavailability).
- **Scalability**: This setup can be easily extended to more nodes by adding their IPs to the `NODE_IPS` array.

This setup provides a robust way to automate the configuration of a PyTorch distributed training environment across multiple nodes, enhancing the efficiency of training large models.