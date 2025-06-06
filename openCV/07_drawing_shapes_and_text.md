OpenCV provides functions to draw shapes like lines, rectangles, and circles, as well as to add text to images. These functions are versatile and customizable.

---

### **1. Drawing Shapes**
The shapes are drawn on an image (either a blank one or an existing image), and the operations directly modify the image.

#### **a. Drawing a Line**
```python
import cv2
import numpy as np

# Create a blank image
image = np.zeros((500, 500, 3), dtype='uint8')

# Draw a line
start_point = (50, 50)
end_point = (450, 50)
color = (255, 0, 0)  # Blue in BGR
thickness = 5
cv2.line(image, start_point, end_point, color, thickness)

cv2.imshow('Line', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

#### **b. Drawing a Rectangle**
```python
# Draw a rectangle
top_left = (100, 100)
bottom_right = (400, 300)
color = (0, 255, 0)  # Green in BGR
thickness = 3  # Use -1 for a filled rectangle
cv2.rectangle(image, top_left, bottom_right, color, thickness)

cv2.imshow('Rectangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

#### **c. Drawing a Circle**
```python
# Draw a circle
center = (250, 250)
radius = 100
color = (0, 0, 255)  # Red in BGR
thickness = -1  # Filled circle
cv2.circle(image, center, radius, color, thickness)

cv2.imshow('Circle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

#### **d. Drawing an Ellipse**
```python
# Draw an ellipse
center = (250, 250)
axes = (150, 100)  # Major and minor axis lengths
angle = 0  # Rotation angle of the ellipse
start_angle = 0  # Starting angle of the arc
end_angle = 180  # Ending angle of the arc
color = (255, 255, 0)  # Cyan in BGR
thickness = 2
cv2.ellipse(image, center, axes, angle, start_angle, end_angle, color, thickness)

cv2.imshow('Ellipse', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

#### **e. Drawing a Polygon**
```python
# Draw a polygon
points = np.array([[100, 300], [200, 200], [300, 300], [250, 400]], dtype=np.int32)
points = points.reshape((-1, 1, 2))  # Reshape for the function
color = (0, 255, 255)  # Yellow in BGR
thickness = 3
cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

cv2.imshow('Polygon', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **2. Adding Text**
You can add text to an image using `cv2.putText`.

```python
# Add text to the image
text = "Hello, OpenCV!"
position = (50, 450)  # Bottom-left corner of the text
font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a font
font_scale = 1.5
color = (255, 255, 255)  # White
thickness = 2

cv2.putText(image, text, position, font, font_scale, color, thickness)

cv2.imshow('Text', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **Customizations**
1. **Colors:** Use the `(B, G, R)` format to define custom colors.
2. **Thickness:** Negative thickness (`-1`) fills shapes.
3. **Font Styles:** OpenCV supports various fonts like:
   - `cv2.FONT_HERSHEY_SIMPLEX`
   - `cv2.FONT_HERSHEY_COMPLEX`
   - `cv2.FONT_HERSHEY_TRIPLEX`

---

### **Example Combining Shapes and Text**
```python
# Create a blank image
image = np.zeros((500, 500, 3), dtype='uint8')

# Draw shapes
cv2.rectangle(image, (50, 50), (450, 450), (0, 255, 0), 3)
cv2.circle(image, (250, 250), 100, (255, 0, 0), -1)
cv2.putText(image, 'OpenCV!', (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow('Shapes and Text', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
``` 

These tools allow you to create custom graphics or annotate images effectively.