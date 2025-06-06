Here’s how to create and display a modal dialog in React Native when a button is clicked.

---

### Step-by-Step Implementation

#### 1. Use the `Modal` Component
React Native provides a built-in `Modal` component to display a dialog.

---

### Full Code Example

```javascript
import React, { useState } from 'react';
import {
  View,
  Text,
  Button,
  Modal,
  StyleSheet,
  TouchableOpacity,
} from 'react-native';

const App = () => {
  const [isModalVisible, setModalVisible] = useState(false);

  // Function to toggle modal visibility
  const toggleModal = () => {
    setModalVisible(!isModalVisible);
  };

  return (
    <View style={styles.container}>
      <Button title="Show Modal" onPress={toggleModal} />

      {/* Modal Component */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={isModalVisible}
        onRequestClose={toggleModal} // Handles back button press
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalText}>This is a modal dialog!</Text>
            <TouchableOpacity
              style={styles.closeButton}
              onPress={toggleModal}
            >
              <Text style={styles.closeButtonText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalOverlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)', // Semi-transparent background
  },
  modalContent: {
    width: 300,
    padding: 20,
    backgroundColor: '#fff',
    borderRadius: 10,
    alignItems: 'center',
    elevation: 5, // Adds shadow for Android
    shadowColor: '#000', // Adds shadow for iOS
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  modalText: {
    fontSize: 18,
    marginBottom: 20,
  },
  closeButton: {
    backgroundColor: '#2196F3',
    padding: 10,
    borderRadius: 5,
  },
  closeButtonText: {
    color: '#fff',
    fontSize: 16,
  },
});

export default App;
```

---

### Explanation

1. **State Management:**
   - `isModalVisible` determines if the modal is visible. Toggled by the `toggleModal` function.

2. **Modal Properties:**
   - `animationType`: Defines how the modal appears (`slide`, `fade`, or `none`).
   - `transparent`: Makes the modal background transparent.
   - `visible`: Controls whether the modal is displayed.
   - `onRequestClose`: Called when the modal is dismissed (e.g., by pressing the back button on Android).

3. **Modal Content:**
   - Styled container with `styles.modalContent` holds the modal text and a "Close" button.
   - The "Close" button hides the modal when pressed.

4. **Overlay:**
   - `styles.modalOverlay` creates a semi-transparent background to focus attention on the modal.

---

### How It Works

1. Tap the "Show Modal" button.
2. The modal dialog slides into view, displaying a message and a "Close" button.
3. Tap "Close" to hide the modal.

---

### Customization

1. **Full-Screen Modal:**
   Remove `transparent` and set the modal container to cover the entire screen.

2. **Custom Animation:**
   Use libraries like `react-native-reanimated` for advanced animations.

3. **Add Inputs:**
   Replace the modal content with text inputs, forms, or interactive elements.

---

This implementation demonstrates how to display and manage modal dialogs in React Native.