# Implementing Convolutional Neural Networks (CNNs) from Scratch

## Overview
Convolutional Neural Networks (CNNs) are a class of deep learning models specifically designed for processing structured grid-like data, such as images. They automatically capture spatial hierarchies and patterns through convolutional layers, making them highly effective in image recognition tasks.

In this section, we will cover the implementation of CNNs from scratch, focusing on key components like convolution layers, pooling layers, and batch normalization.

## Key Components of CNNs

### 1. **Convolutional Layer (Conv2D)**
   - The core operation in CNNs is the convolutional layer, which applies a set of filters to the input data. Each filter performs a dot product between its weights and the input, producing a feature map.
   - **Input**: A 3D matrix (height, width, depth) representing an image.
   - **Output**: Feature maps that emphasize specific aspects of the input (e.g., edges, textures).
   - **Formula**: 
     \[
     \text{Feature Map} = (Input * Filter) + Bias
     \]
     where `*` denotes convolution.

   ```python
   def conv2d(input, filters, stride=1, padding=0):
       # Apply padding to input if necessary
       if padding > 0:
           input = np.pad(input, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

       # Output dimensions
       output_height = (input.shape[0] - filters.shape[0]) // stride + 1
       output_width = (input.shape[1] - filters.shape[1]) // stride + 1
       output_depth = filters.shape[3]
       output = np.zeros((output_height, output_width, output_depth))

       # Perform convolution
       for i in range(0, output_height):
           for j in range(0, output_width):
               for k in range(output_depth):
                   region = input[i*stride:i*stride+filters.shape[0], j*stride:j*stride+filters.shape[1], :]
                   output[i, j, k] = np.sum(region * filters[:, :, :, k])

       return output
   ```

### 2. **Pooling Layer**
   - The pooling layer reduces the spatial dimensions (height and width) of the feature maps while retaining the most important features. Common pooling techniques include max pooling and average pooling.
   - **Max Pooling**: Selects the maximum value from a set of neighboring pixels, providing down-sampled representations of the input.
   - **Formula (Max Pooling)**:
     \[
     \text{Max Pool Output} = \max(\text{Local Region})
     \]
   ```python
   def max_pooling(input, pool_size=2, stride=2):
       output_height = (input.shape[0] - pool_size) // stride + 1
       output_width = (input.shape[1] - pool_size) // stride + 1
       output_depth = input.shape[2]
       output = np.zeros((output_height, output_width, output_depth))

       # Perform max pooling
       for i in range(0, output_height):
           for j in range(0, output_width):
               for k in range(output_depth):
                   region = input[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size, k]
                   output[i, j, k] = np.max(region)

       return output
   ```

### 3. **Batch Normalization**
   - Batch Normalization (BN) normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation. It helps in stabilizing and speeding up the training process.
   - **Formula**:
     \[
     \text{BN}(x) = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma_{\text{batch}}^2 + \epsilon}}
     \]
     where \(\mu_{\text{batch}}\) is the batch mean, \(\sigma_{\text{batch}}^2\) is the batch variance, and \(\epsilon\) is a small constant to avoid division by zero.

   ```python
   def batch_normalization(input, gamma, beta, epsilon=1e-5):
       mean = np.mean(input, axis=0)
       variance = np.var(input, axis=0)
       normalized = (input - mean) / np.sqrt(variance + epsilon)
       return gamma * normalized + beta
   ```

## Steps to Implement a CNN from Scratch

1. **Define the Network Architecture**:
   - Choose the number of layers (conv layers, pooling layers, and fully connected layers).
   - Decide on the filter size, stride, padding, and activation functions.

2. **Implement Forward Propagation**:
   - Pass the input through the layers (convolution -> activation -> pooling).
   - Apply batch normalization where required.
   
3. **Loss Function and Backpropagation**:
   - Compute the loss using a loss function like cross-entropy for classification tasks.
   - Perform backpropagation to update the weights of the filters.

4. **Train the Model**:
   - Use an optimization algorithm like stochastic gradient descent (SGD) to minimize the loss.
   - Iterate over the training dataset for multiple epochs.

5. **Test and Validate**:
   - Evaluate the performance of the trained model on test data to check for generalization.

## Sample CNN Architecture

Here’s a basic CNN architecture:
- Input: 28x28x1 (grayscale image)
- Conv Layer 1: 32 filters (3x3), ReLU activation
- Max Pooling Layer 1: 2x2 pool size
- Conv Layer 2: 64 filters (3x3), ReLU activation
- Max Pooling Layer 2: 2x2 pool size
- Fully Connected Layer: 128 neurons, ReLU activation
- Output Layer: Softmax activation for classification
