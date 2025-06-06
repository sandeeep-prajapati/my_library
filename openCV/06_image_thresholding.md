### **Image Thresholding**
Image thresholding is a technique in image processing used to separate objects or regions of interest in an image from the background. It involves setting a threshold value, and any pixel value above or below that threshold is categorized into one of two groups, effectively creating a binary image.

---

### **How It Works**
For a grayscale image:
- Pixels with intensity **greater than or equal to** the threshold are set to a maximum value (white, typically 255).
- Pixels with intensity **less than** the threshold are set to a minimum value (black, typically 0).

This converts the image into a binary format (black and white).

---

### **Types of Thresholding**
1. **Simple Thresholding**
   - A fixed threshold value is manually chosen.

2. **Adaptive Thresholding**
   - The threshold value varies over the image based on local properties.

3. **Otsu's Thresholding**
   - Automatically determines an optimal threshold value.

---

### **Simple Thresholding**
```python
import cv2

# Load a grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply simple thresholding
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('Binary Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
- `127` is the threshold value.
- `255` is the maximum value for white pixels.
- `cv2.THRESH_BINARY` applies the binary threshold.

---

### **Adaptive Thresholding**
Adaptive thresholding calculates the threshold for small regions of the image, making it useful for images with varying lighting conditions.

```python
# Apply adaptive thresholding
binary_image = cv2.adaptiveThreshold(image, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)

cv2.imshow('Adaptive Binary Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
- `11` is the block size (size of the neighborhood to calculate the threshold).
- `2` is the constant subtracted from the mean or weighted mean.

---

### **Otsu's Thresholding**
Otsu's method calculates the optimal threshold value automatically by minimizing intra-class variance.

```python
# Apply Otsu's thresholding
_, binary_image = cv2.threshold(image, 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('Otsu Binary Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
- The threshold value is ignored and set automatically by Otsu's algorithm.

---

### **Applications of Binary Images**
- **Edge detection**
- **Object segmentation**
- **Document scanning**
- **License plate recognition**

Thresholding is often a preprocessing step for more complex computer vision tasks.