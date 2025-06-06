Automating memory profiling of PyTorch models during training can help you identify memory usage patterns and optimize resource allocation. Below is a Bash script that sets up a PyTorch training environment, runs the training process while profiling memory usage, and saves the profiling results for analysis.

### Bash Script for Memory Profiling in PyTorch

**`memory_profiling.sh`**:

```bash
#!/bin/bash

# Configuration variables
USER="ubuntu"  # Change if using a different AMI user
INSTANCE_TYPE="p2.xlarge"  # Use a suitable instance type
KEY_NAME="your-key-pair"     # Your SSH key pair
AMI_ID="ami-XXXXXXXX"        # Replace with your AMI ID
SECURITY_GROUP="your-security-group"  # Replace with your security group
INSTANCE_ID=""
TRAINING_SCRIPT_URL="https://your-url.com/train.py"  # URL to your training script
PROFILING_SCRIPT_URL="https://your-url.com/profile.py"  # URL to your profiling script

# Launch an EC2 instance if it is not already running
if [ -z "$INSTANCE_ID" ]; then
    echo "Launching EC2 instance..."
    INSTANCE_ID=$(aws ec2 run-instances --image-id $AMI_ID --count 1 --instance-type $INSTANCE_TYPE --key-name $KEY_NAME --security-group-ids $SECURITY_GROUP --query 'Instances[0].InstanceId' --output text)
    echo "Instance launched with ID: $INSTANCE_ID"
    
    # Wait until the instance is running
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID
fi

# Get the public IP of the instance
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Connecting to instance at $PUBLIC_IP..."

# SSH into the instance and set up the environment
ssh -o StrictHostKeyChecking=no -i "$KEY_NAME.pem" "$USER@$PUBLIC_IP" << EOF
    echo "Setting up the environment..."
    
    # Update and install necessary packages
    sudo apt-get update
    sudo apt-get install -y python3-pip

    # Install PyTorch and torchvision (modify as necessary)
    pip3 install torch torchvision

    # Install memory profiler
    pip3 install memory-profiler

    # Create a directory for the project
    mkdir -p ~/pytorch_training
    cd ~/pytorch_training

    # Download the training and profiling scripts
    echo "Downloading training script..."
    wget $TRAINING_SCRIPT_URL -O train.py
    echo "Downloading profiling script..."
    wget $PROFILING_SCRIPT_URL -O profile.py

    # Start memory profiling
    echo "Starting memory profiling..."
    mprof run python3 profile.py
EOF

# Print completion message
echo "Memory profiling initiated on EC2 instance: $INSTANCE_ID"
```

### Profiling Script for PyTorch Model Training

Create a Python script named `profile.py` that will handle the training of the model while also profiling memory usage.

**`profile.py`**:

```python
import argparse
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from memory_profiler import profile

@profile
def train_model(epochs, learning_rate):
    # Define data transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # Load dataset (adjust based on your dataset)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Load model
    model = models.resnet18(num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:  # Log every 100 batches
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Save the final model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as model.pth.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PyTorch model with memory profiling.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    train_model(args.epochs, args.learning_rate)
```

### Explanation of the Script Components

1. **Bash Script Setup**:
   - The script launches an EC2 instance and sets up the environment, installing Python, PyTorch, and the `memory-profiler` library.
   - It downloads the training and profiling scripts to the EC2 instance.
   - It uses `mprof run` to execute the profiling script, which will monitor memory usage during the training process.

2. **Memory Profiling in Python**:
   - The `profile.py` script contains the training loop and is decorated with `@profile` from the `memory_profiler` library, allowing you to monitor memory usage for each line in the decorated function.
   - The model, optimizer, and data loading processes are defined, and the training loop logs the loss periodically.

### Step 1: Make the Bash Script Executable

Make the Bash script executable by running:

```bash
chmod +x memory_profiling.sh
```

### Step 2: Run the Bash Script

Execute the script to initiate the EC2 setup and memory profiling:

```bash
./memory_profiling.sh
```

### Additional Considerations

- **Instance Termination**: Consider adding functionality to terminate the EC2 instance after profiling is complete to avoid incurring additional costs.
- **Output Files**: The profiling results will be saved in a `.dat` file by `memory-profiler`. You can visualize this using `mprof plot`.
- **Monitoring and Logging**: Implement additional logging to track performance metrics during training.

This setup provides an automated way to profile memory usage during the training of PyTorch models on AWS EC2.