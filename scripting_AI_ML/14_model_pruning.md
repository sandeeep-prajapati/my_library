To automate PyTorch model pruning using a Batch script, you'll need to follow a few steps. This involves creating a Python script that performs the pruning of the model and then writing a Batch script to execute that Python script. Below is a guide to help you through the process.

### Step 1: Create a Python Script for Model Pruning

First, create a Python script that implements model pruning using PyTorch. The script will load a pre-trained model, apply pruning, and save the pruned model for deployment.

**`prune_model.py`**:

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
import sys

def prune_model(model, pruning_ratio):
    # Apply global unstructured pruning to all linear layers
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
    return model

def save_model(model, output_path):
    # Remove pruning from model
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, 'weight')
    
    # Save the pruned model
    torch.save(model.state_dict(), output_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prune_model.py <model_path> <output_path> <pruning_ratio>")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]
    pruning_ratio = float(sys.argv[3])

    # Load pre-trained model
    model = models.resnet18()  # Example: Using ResNet-18, replace with your model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prune the model
    pruned_model = prune_model(model, pruning_ratio)

    # Save the pruned model
    save_model(pruned_model, output_path)
    print(f"Pruned model saved to {output_path}.")
```

### Step 2: Create the Batch Script to Automate Pruning

Next, create a Batch script that runs the Python pruning script. This Batch script will take the model path, output path, and pruning ratio as arguments.

**`run_pruning.bat`**:

```batch
@echo off
REM Check command-line arguments
IF "%~3"=="" (
    echo Usage: %0 ^<model_path^> ^<output_path^> ^<pruning_ratio^>
    exit /b 1
)

SET MODEL_PATH=%~1
SET OUTPUT_PATH=%~2
SET PRUNING_RATIO=%~3

REM Check if the model file exists
IF NOT EXIST "%MODEL_PATH%" (
    echo Error: Model file does not exist at %MODEL_PATH%
    exit /b 1
)

REM Run the pruning script
python prune_model.py "%MODEL_PATH%" "%OUTPUT_PATH%" "%PRUNING_RATIO%"

REM Check if pruning was successful
IF ERRORLEVEL 1 (
    echo Error during model pruning!
    exit /b 1
)

echo Pruning completed successfully! Pruned model saved to %OUTPUT_PATH%.
```

### Step 3: Execute the Batch Script

To run the Batch script, open a command prompt and navigate to the directory containing your scripts. You can run the script as follows:

```batch
run_pruning.bat path\to\model.pth path\to\pruned_model.pth 0.5
```

### Explanation of Key Components

1. **Python Script (`prune_model.py`)**:
   - Loads a pre-trained model (e.g., ResNet-18).
   - Applies global unstructured pruning to all linear layers based on the specified pruning ratio.
   - Saves the pruned model to the specified output path.

2. **Batch Script (`run_pruning.bat`)**:
   - Validates command-line arguments to ensure that the necessary parameters are provided.
   - Checks if the model file exists before proceeding.
   - Executes the Python script with the provided parameters and checks for any errors during execution.

### Additional Considerations

- **Model Compatibility**: Ensure that the model you are using is compatible with the pruning operations. You can modify the pruning function to handle different model architectures.

- **Pruning Ratio**: The pruning ratio indicates the proportion of parameters to prune. Adjust it according to your needs and testing.

- **Dependencies**: Make sure you have installed the required libraries. You can install them using:

  ```bash
  pip install torch torchvision
  ```

- **Model Evaluation**: After pruning, you might want to fine-tune or evaluate the pruned model to assess its performance on your specific tasks.

This setup allows you to automate the process of pruning a PyTorch model for mobile deployment using a Batch script, making it easier to reduce the model size while maintaining efficiency.