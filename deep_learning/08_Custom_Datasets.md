# 8. **Custom Datasets**

## How to Create Custom Datasets and Manage Transforms Using `torchvision.transforms`

In many deep learning tasks, you may need to work with datasets that are not readily available in PyTorch’s built-in libraries. PyTorch makes it easy to create custom datasets by extending the `torch.utils.data.Dataset` class. Additionally, the `torchvision.transforms` module provides a variety of tools for applying preprocessing and data augmentation to your dataset.

### 8.1 **Creating a Custom Dataset**

To create a custom dataset, you need to subclass `torch.utils.data.Dataset` and define two essential methods:
- `__len__`: Returns the size of the dataset.
- `__getitem__`: Retrieves the sample (data point) and its corresponding label from the dataset at the given index.

Here’s an example of how to create a custom dataset from image files:

- **Example: Custom Image Dataset**:
  ```python
  from torch.utils.data import Dataset
  from PIL import Image

  class CustomImageDataset(Dataset):
      def __init__(self, image_paths, labels, transform=None):
          self.image_paths = image_paths
          self.labels = labels
          self.transform = transform

      def __len__(self):
          return len(self.image_paths)

      def __getitem__(self, idx):
          image = Image.open(self.image_paths[idx])
          label = self.labels[idx]
          
          if self.transform:
              image = self.transform(image)
          
          return image, label
  ```

- **Parameters**:
  - `image_paths`: A list of paths to the images.
  - `labels`: The corresponding labels for the images.
  - `transform`: Optional transformations (data augmentation, normalization, etc.) applied to the images.

- **Use Case**: This approach is ideal for custom image datasets stored locally on your machine, whether they are images, text files, or any other data types.

### 8.2 **Managing Transforms Using `torchvision.transforms`**

The `torchvision.transforms` module is a powerful utility for data preprocessing and augmentation. It provides a variety of operations to convert raw data into forms suitable for training deep learning models.

#### 8.2.1 **Basic Transforms**

- **ToTensor**: Converts a PIL image or NumPy array to a PyTorch tensor.
  
  ```python
  from torchvision import transforms

  transform = transforms.ToTensor()
  ```

- **Normalize**: Normalizes a tensor image by subtracting the mean and dividing by the standard deviation.
  
  ```python
  transform = transforms.Normalize(mean=[0.5], std=[0.5])
  ```

- **Compose**: Combines multiple transforms into one.
  
  ```python
  transform = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])
  ```

#### 8.2.2 **Data Augmentation with Transforms**

- **Resize**: Resizes the image to the given size.
  ```python
  transforms.Resize((128, 128))
  ```

- **RandomHorizontalFlip**: Randomly flips the image horizontally.
  ```python
  transforms.RandomHorizontalFlip(p=0.5)
  ```

- **RandomCrop**: Randomly crops the image to a specified size.
  ```python
  transforms.RandomCrop((100, 100))
  ```

- **RandomRotation**: Randomly rotates the image by a given degree.
  ```python
  transforms.RandomRotation(degrees=45)
  ```

#### 8.2.3 **Using Transforms in a Custom Dataset**

You can pass these transformations when initializing a custom dataset. They will be applied to the data points as they are fetched.

- **Example: Applying Transforms to a Custom Dataset**:
  ```python
  from torchvision import transforms

  transform = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])

  dataset = CustomImageDataset(image_paths, labels, transform=transform)
  ```

### 8.3 **Combining Custom Datasets with `DataLoader`**

Once you've defined your custom dataset and applied the necessary transforms, you can easily use it with `DataLoader` for batch processing, shuffling, and parallel data loading.

- **Example**:
  ```python
  from torch.utils.data import DataLoader

  dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

  for images, labels in dataloader:
      # Your training loop here
      pass
  ```

### 8.4 **Handling Other Data Types (Text, CSV, etc.)**

You can also create custom datasets for non-image data types, such as text or CSV files. The same principles apply:
1. **Load the data**: Read the data from its source (CSV, text files, etc.).
2. **Preprocess the data**: Apply any necessary transformations.
3. **Return data and labels**: Format the data into tensors and return them.

- **Example: Custom Dataset for Text Data**:
  ```python
  class CustomTextDataset(Dataset):
      def __init__(self, text_data, labels, transform=None):
          self.text_data = text_data
          self.labels = labels
          self.transform = transform

      def __len__(self):
          return len(self.text_data)

      def __getitem__(self, idx):
          text = self.text_data[idx]
          label = self.labels[idx]

          if self.transform:
              text = self.transform(text)
          
          return text, label
  ```

### 8.5 **Best Practices for Custom Datasets**

- **Efficiency**: When loading large datasets, avoid loading all the data into memory at once. Instead, load data on-the-fly in the `__getitem__` method to save memory.
- **Transforms**: Use transformations such as normalization and data augmentation during the training process to improve model generalization.
- **Parallel Loading**: Use multiple workers (`num_workers > 0`) in the `DataLoader` to improve data loading efficiency, especially for large datasets.

---

This section provides a comprehensive overview of how to create and work with custom datasets in PyTorch, including applying transformations using `torchvision.transforms`, which is crucial for efficient preprocessing and augmentation of data during model training.