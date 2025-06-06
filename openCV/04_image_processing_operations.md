OpenCV is a powerful library for image processing, and it supports various basic operations like resizing, cropping, and rotating. Below is an overview of these operations:

---

### **1. Resizing**
Resizing changes the dimensions of an image while preserving its content.

#### **Code Example:**
```python
import cv2

# Load an image
image = cv2.imread('image.jpg')

# Resize the image to specific dimensions
resized_image = cv2.resize(image, (width, height))  # e.g., (300, 300)

# Resize while preserving the aspect ratio
aspect_ratio = image.shape[1] / image.shape[0]
new_width = 300
new_height = int(new_width / aspect_ratio)
resized_aspect = cv2.resize(image, (new_width, new_height))

cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **2. Cropping**
Cropping extracts a specific region of interest (ROI) from an image.

#### **Code Example:**
```python
# Define the region of interest (ROI)
# Format: image[y1:y2, x1:x2]
cropped_image = image[50:200, 100:300]

cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **3. Rotating**
Rotating an image involves turning it around its center by a specified angle.

#### **Code Example:**
```python
# Get the image dimensions
(h, w) = image.shape[:2]

# Define the center of the image
center = (w // 2, h // 2)

# Define a rotation matrix (angle in degrees, clockwise)
angle = 45  # Rotate 45 degrees
scale = 1.0  # Scaling factor
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# Perform the rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **Other Useful Operations**
- **Flipping:**
  ```python
  flipped_image = cv2.flip(image, flipCode=1)  # flipCode: 0 (vertical), 1 (horizontal), -1 (both)
  ```
- **Translation:**
  ```python
  # Shift the image by (x, y)
  translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
  translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
  ```
- **Scaling:**
  ```python
  scaled_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
  ```

These operations form the foundation for many complex image-processing tasks.