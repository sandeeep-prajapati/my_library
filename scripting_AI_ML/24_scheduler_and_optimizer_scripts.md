Automating the use of PyTorch learning rate schedulers and optimizers can streamline the training process, allowing you to easily adjust hyperparameters and monitor performance. Below is a Bash script that sets up a PyTorch training environment, runs the training with specified optimizers and learning rate schedulers, and logs the results.

### Bash Script for Automating Learning Rate Schedulers and Optimizers

**`automate_lr_scheduler.sh`**:

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

    # Create a directory for the project
    mkdir -p ~/pytorch_training
    cd ~/pytorch_training

    # Download the training script
    echo "Downloading training script..."
    wget $TRAINING_SCRIPT_URL -O train.py

    # Start training with specified optimizers and schedulers
    echo "Starting training with learning rate scheduler and optimizer..."
    python3 train.py --optimizer adam --scheduler step_lr --epochs 10 --lr 0.001
EOF

# Print completion message
echo "Training initiated on EC2 instance: $INSTANCE_ID"
```

### Training Script with Optimizers and Learning Rate Schedulers

Create a Python script named `train.py` that will handle the training of the model using specified optimizers and learning rate schedulers.

**`train.py`**:

```python
import argparse
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

def get_optimizer(optimizer_name, model, learning_rate):
    if optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_scheduler(scheduler_name, optimizer):
    if scheduler_name == 'step_lr':
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif scheduler_name == 'exponential_lr':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

def train_model(optimizer_name, scheduler_name, epochs, learning_rate):
    # Define data transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # Load dataset (adjust based on your dataset)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Load model
    model = models.resnet18(num_classes=10)
    optimizer = get_optimizer(optimizer_name, model, learning_rate)
    scheduler = get_scheduler(scheduler_name, optimizer)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        
        scheduler.step()  # Update learning rate
        print(f"Epoch {epoch + 1}/{epochs}, Learning Rate: {scheduler.get_last_lr()}, Loss: {loss.item():.4f}")

    # Save the final model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as model.pth.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PyTorch model with learning rate scheduler and optimizer.")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='step_lr', choices=['step_lr', 'exponential_lr'], help='Learning rate scheduler to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    train_model(args.optimizer, args.scheduler, args.epochs, args.lr)
```

### Explanation of the Script Components

1. **Bash Script Setup**:
   - The script launches an EC2 instance and sets up the environment, installing Python and PyTorch.
   - It downloads the training script to the EC2 instance and initiates training with specified optimizer and scheduler parameters.

2. **Training Script**:
   - The `train.py` script defines functions to select the optimizer and learning rate scheduler based on user inputs.
   - The training loop logs the learning rate and loss for each epoch, allowing you to monitor performance.

### Step 1: Make the Bash Script Executable

Make the Bash script executable by running:

```bash
chmod +x automate_lr_scheduler.sh
```

### Step 2: Run the Bash Script

Execute the script to initiate the EC2 setup and training:

```bash
./automate_lr_scheduler.sh
```

### Additional Considerations

- **Instance Termination**: You may want to add functionality to terminate the EC2 instance after training to avoid additional costs.
- **Custom Dataset**: Modify the data loading section in the `train.py` script to use a custom dataset as needed.
- **Monitoring and Logging**: Implement additional logging to track performance metrics or save logs to a file for further analysis.

This setup provides an automated way to train PyTorch models using different optimizers and learning rate schedulers on AWS EC2.