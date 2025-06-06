Here’s a Bash script to automate data augmentation on image datasets using PyTorch’s `torchvision.transforms`. This script will apply a set of common augmentations to each image in the dataset and save the augmented images in a new folder.

### Prerequisites

1. **Install PyTorch and torchvision** if not already installed:

   ```bash
   pip install torch torchvision
   ```

2. **Folder Structure**: The original images should be organized in folders by category (e.g., `dataset/class1`, `dataset/class2`, etc.).

### Bash Script: `augment_images.sh`

This script:
1. Loops through each category folder.
2. Applies augmentations to each image using a Python script.
3. Saves augmented images in a new folder structure mirroring the original.

```bash
#!/bin/bash

# Define directories
DATASET_DIR="./dataset"  # Original dataset directory
OUTPUT_DIR="./augmented_dataset"  # Directory to save augmented images

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through each class in the dataset
for CLASS_DIR in "$DATASET_DIR"/*; do
    if [ -d "$CLASS_DIR" ]; then
        CLASS_NAME=$(basename "$CLASS_DIR")
        echo "Processing class: $CLASS_NAME"

        # Create corresponding class directory in output folder
        mkdir -p "$OUTPUT_DIR/$CLASS_NAME"

        # Run the Python script for each class
        python3 - <<END
import os
import torch
from torchvision import transforms
from PIL import Image

# Define input and output paths
input_dir = "$CLASS_DIR"
output_dir = os.path.join("$OUTPUT_DIR", "$CLASS_NAME")

# Data augmentations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
])

# Apply augmentations and save images
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    with Image.open(img_path) as img:
        augmented_img = transform(img)
        
        # Save augmented image
        augmented_img.save(os.path.join(output_dir, f"aug_{img_name}"))

print(f"Augmentation complete for class: {input_dir}")
END

        echo "Augmentation done for class: $CLASS_NAME"
        echo
    fi
done

echo "All augmentations completed!"
```

### Explanation of the Bash Script

1. **Define Directories**: 
   - `DATASET_DIR`: The original dataset directory containing class subdirectories.
   - `OUTPUT_DIR`: The target directory for augmented images.

2. **Loop Through Each Class Directory**: 
   - Checks if the path is a directory.
   - Creates a corresponding directory in `OUTPUT_DIR` for each class.

3. **Python Augmentation Code (Inside the Bash Script)**:
   - Uses an inline Python script within the Bash script to apply augmentations:
     - **`RandomHorizontalFlip`**: Randomly flips images horizontally with a 50% probability.
     - **`RandomRotation`**: Rotates images within a ±15-degree range.
     - **`ColorJitter`**: Randomly changes the brightness, contrast, and saturation.
     - **`RandomResizedCrop`**: Crops images to a specified size with random scaling.
   - The Python script then saves each augmented image with a prefix (`aug_`) in the output directory.

4. **Output Messages**: The Bash script provides feedback for each class processed.

### Running the Script

1. Save this script as `augment_images.sh`.
2. Make it executable:

   ```bash
   chmod +x augment_images.sh
   ```

3. Run the script:

   ```bash
   ./augment_images.sh
   ```

This will create an `augmented_dataset` directory with augmented images organized by class, ready for use in training.