Geometric transformations are operations that modify the geometry of an image, such as shifting its position (translation), rotating it, or resizing it (scaling). These transformations are essential for various image processing tasks like image alignment, object detection, and augmenting datasets. OpenCV provides functions for these transformations, and they are usually represented using transformation matrices.

---

### **1. Translation**
Translation shifts the image by a certain distance along the X and Y axes. In OpenCV, translation is performed using a translation matrix.

#### **Translation Matrix:**
The translation matrix for a 2D image is:

\[
T = \begin{bmatrix} 1 & 0 & Tx \\ 0 & 1 & Ty \\ 0 & 0 & 1 \end{bmatrix}
\]

Where:
- \(Tx\) and \(Ty\) are the translation distances along the X and Y axes.

#### **Code Example:**
```python
import cv2
import numpy as np

# Load the image
image = cv2.imread('image.jpg')

# Get the image dimensions
rows, cols = image.shape[:2]

# Define the translation matrix (Tx, Ty)
Tx = 100  # Translate 100 pixels in X direction
Ty = 50   # Translate 50 pixels in Y direction
M = np.float32([[1, 0, Tx], [0, 1, Ty]])

# Apply the translation
translated_image = cv2.warpAffine(image, M, (cols, rows))

# Display the result
cv2.imshow('Translated Image', translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **2. Rotation**
Rotation involves rotating the image by a specified angle around a center point. The rotation matrix is used to perform the rotation.

#### **Rotation Matrix:**
The rotation matrix is defined as:

\[
R = \begin{bmatrix} \cos(\theta) & -\sin(\theta) & (1 - \cos(\theta)) \cdot C_x + \sin(\theta) \cdot C_y \\ \sin(\theta) & \cos(\theta) & (1 - \cos(\theta)) \cdot C_y - \sin(\theta) \cdot C_x \\ 0 & 0 & 1 \end{bmatrix}
\]

Where:
- \(\theta\) is the rotation angle.
- \(C_x, C_y\) are the coordinates of the center of rotation.

#### **Code Example:**
```python
# Define the center of the image (the point of rotation)
center = (cols // 2, rows // 2)

# Define the rotation matrix
angle = 45  # Rotate by 45 degrees
scale = 1.0  # No scaling
M = cv2.getRotationMatrix2D(center, angle, scale)

# Apply the rotation
rotated_image = cv2.warpAffine(image, M, (cols, rows))

# Display the result
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **3. Scaling**
Scaling involves resizing an image by a specified factor along the X and Y axes, either shrinking or enlarging the image.

#### **Scaling Matrix:**
Scaling can be done with the following scaling matrix:

\[
S = \begin{bmatrix} S_x & 0 & 0 \\ 0 & S_y & 0 \\ 0 & 0 & 1 \end{bmatrix}
\]

Where:
- \(S_x\) and \(S_y\) are the scaling factors along the X and Y axes.

Alternatively, scaling can be done without creating a matrix explicitly by using the `cv2.resize()` function.

#### **Code Example:**
```python
# Scaling the image by a factor of 2
scaled_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# Display the result
cv2.imshow('Scaled Image', scaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **4. Affine Transformation**
Affine transformation is a combination of translation, rotation, scaling, and shearing. It preserves parallelism (i.e., straight lines remain straight). It can be represented by a 2x3 matrix.

\[
A = \begin{bmatrix} a & b & Tx \\ c & d & Ty \end{bmatrix}
\]

Where:
- \(a, b, c, d\) control the linear transformation (scaling, rotation, shearing).
- \(Tx\) and \(Ty\) represent translation.

#### **Code Example:**
```python
# Define points for affine transformation (3 points)
points1 = np.float32([[50, 50], [200, 50], [50, 200]])
points2 = np.float32([[10, 100], [250, 50], [100, 250]])

# Compute the affine transformation matrix
M = cv2.getAffineTransform(points1, points2)

# Apply the affine transformation
affine_image = cv2.warpAffine(image, M, (cols, rows))

# Display the result
cv2.imshow('Affine Transformed Image', affine_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **5. Perspective Transformation (Homography)**
Perspective transformation involves mapping points from one plane to another, creating a perspective distortion effect. This is often used for tasks like "birds-eye view" of a scene.

#### **Code Example:**
```python
# Define four points in the source image
pts1 = np.float32([[100, 100], [200, 100], [100, 200], [200, 200]])

# Define four points in the destination image
pts2 = np.float32([[50, 150], [250, 100], [50, 250], [250, 250]])

# Compute the perspective transformation matrix
M = cv2.getPerspectiveTransform(pts1, pts2)

# Apply the perspective transformation
perspective_image = cv2.warpPerspective(image, M, (cols, rows))

# Display the result
cv2.imshow('Perspective Transformed Image', perspective_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **Summary of Common Geometric Transformations in OpenCV:**
1. **Translation**: Shifts the image along the X and Y axes.
2. **Rotation**: Rotates the image by a specific angle around a point.
3. **Scaling**: Resizes the image by a given factor.
4. **Affine Transformation**: Combination of translation, rotation, scaling, and shearing.
5. **Perspective Transformation**: Mapped points to create a perspective effect.

These transformations are widely used in computer vision for tasks like image alignment, object tracking, and image augmentation for machine learning models.