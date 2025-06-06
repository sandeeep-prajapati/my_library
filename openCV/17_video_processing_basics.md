To load, display, and process video files in OpenCV, you can use `cv2.VideoCapture` to load the video and `cv2.imshow` to display it. Basic video processing includes frame-by-frame extraction, resizing, and applying filters. Here's a guide on how to load, display, and process videos using OpenCV:

### 1. Loading a Video
Use `cv2.VideoCapture()` to load a video file or a live video stream.

```python
import cv2

# Load a video file (replace 'video.mp4' with your video file path)
cap = cv2.VideoCapture('video.mp4')

# Check if the video was loaded successfully
if not cap.isOpened():
    print("Error: Could not open video.")
```

### 2. Reading Frames from the Video
Use `cap.read()` to read frames from the video. It returns a boolean value indicating if the frame was read successfully, and the frame itself.

```python
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or error reading frame.")
        break
    
    # Process the frame (e.g., resizing or applying filters)
    frame = cv2.resize(frame, (640, 480))  # Resize frame to 640x480

    # Display the frame
    cv2.imshow('Video Frame', frame)
    
    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
```

### 3. Writing Frames to a New Video File
To save the processed video, you can use `cv2.VideoWriter`.

```python
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Write the frame to the output video file
    out.write(frame)
    
    # Display the frame
    cv2.imshow('Video Frame', frame)
    
    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
```

### 4. Basic Operations for Video Handling
Some common operations when working with videos include:

- **Resizing frames**: Use `cv2.resize(frame, (width, height))` to resize the frame to a specific resolution.
  
- **Converting to grayscale**: Use `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` to convert each frame to grayscale.

- **Applying filters**: You can apply filters like Gaussian blur using `cv2.GaussianBlur(frame, (5, 5), 0)`.

- **Frame processing**: For more complex processing, you can detect edges, track objects, or apply any computer vision algorithms on the frames.

### Example: Grayscale Video
Here's an example where each frame is converted to grayscale before displaying:

```python
import cv2

cap = cv2.VideoCapture('video.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Display the grayscale frame
    cv2.imshow('Grayscale Video', gray_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 5. Handling Video Properties
You can get properties of the video like the frame rate, width, and height using `cap.get()`:

```python
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

print(f"Width: {frame_width}, Height: {frame_height}, FPS: {frame_rate}")
```

### Conclusion
In summary, OpenCV provides a simple way to load, process, and save videos frame-by-frame. You can resize, apply filters, or even perform more complex tasks such as object detection on each frame.