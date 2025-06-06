To fetch and display an image from a remote URL in a React Native app, you can use the built-in `<Image>` component provided by React Native. This component allows you to load images from various sources, including remote URLs.

Here’s a step-by-step guide to achieve this:

### **1. Basic Setup**

Ensure you have a React Native environment set up. If not, you can refer to the official [React Native documentation](https://reactnative.dev/docs/environment-setup).

### **2. Code Example**

```javascript
import React from 'react';
import { View, Image, StyleSheet } from 'react-native';

const App = () => {
  const imageUrl = 'https://your-image-url-here.com/image.jpg'; // Replace with the URL of your image

  return (
    <View style={styles.container}>
      <Image
        source={{ uri: imageUrl }} // Specify the URL of the image to fetch
        style={styles.image} // Apply styles to the image
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
  },
  image: {
    width: 300,  // Set the width of the image
    height: 300, // Set the height of the image
    resizeMode: 'contain', // Adjust how the image fits within the given width and height
  },
});

export default App;
```

### **3. Explanation of the Code:**

- **`<Image>` Component:**
  - The `source` prop accepts a URL as an object with the `uri` key (`{ uri: 'image-url' }`).
  - This tells React Native to load the image from the remote server.
  
- **Styles:**
  - `resizeMode: 'contain'` ensures the image maintains its aspect ratio while fitting within the given `width` and `height`. Other options include `'cover'`, `'stretch'`, and `'center'`.
  - The `image` style sets the image dimensions. Adjust these values to fit the image appropriately within your app's layout.

- **Container:**
  - The `container` style ensures the image is centered both vertically and horizontally using `justifyContent: 'center'` and `alignItems: 'center'`.

### **4. Running the App:**

Once the code is written, you can run the app on your device or emulator:

For **iOS** (on macOS):
```bash
npx react-native run-ios
```

For **Android**:
```bash
npx react-native run-android
```

### **5. Notes:**
- Ensure the remote image URL is publicly accessible (i.e., the image is hosted on a server that does not require authentication or CORS headers).
- If you encounter performance issues while loading large images, you can consider caching images or using libraries like [react-native-fast-image](https://github.com/DylanVann/react-native-fast-image) for optimized image loading.

This is a basic way to fetch and display an image from a remote URL in React Native. You can customize this further by adding loading indicators, error handling, or other UI elements as needed.