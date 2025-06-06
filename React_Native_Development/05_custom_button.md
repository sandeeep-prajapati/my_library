To create a reusable custom button component in React Native that accepts props for styling and text, you can follow the steps below:

### 1. Create the Button Component

Start by creating a new file for the button component. For example, you can create a file named `CustomButton.js`.

### 2. CustomButton.js

```javascript
import React from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';

// Custom Button Component
const CustomButton = ({ title, onPress, buttonStyle, textStyle }) => {
  return (
    <TouchableOpacity style={[styles.button, buttonStyle]} onPress={onPress}>
      <Text style={[styles.text, textStyle]}>{title}</Text>
    </TouchableOpacity>
  );
};

// Default Styles for Button and Text
const styles = StyleSheet.create({
  button: {
    backgroundColor: '#007bff',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 5,
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default CustomButton;
```

### 3. Explanation:
- **Props**:
  - `title`: The text to be displayed on the button.
  - `onPress`: The function to be executed when the button is pressed.
  - `buttonStyle`: Custom styles for the button (optional).
  - `textStyle`: Custom styles for the text inside the button (optional).
- **`TouchableOpacity`**: This component is used to create a touchable button.
- **`StyleSheet`**: Used to define default styles for the button and text, with flexibility to override these styles using props.

### 4. Use the CustomButton Component in App.js

Now, integrate the `CustomButton` into the `App.js` file or any other component where you want to use it.

### 5. App.js

```javascript
import React from 'react';
import { View, StyleSheet } from 'react-native';
import CustomButton from './CustomButton';  // Import the custom button component

const App = () => {
  // Handle button press
  const handlePress = () => {
    alert('Button Pressed!');
  };

  return (
    <View style={styles.container}>
      <CustomButton
        title="Default Button"
        onPress={handlePress}
      />
      <CustomButton
        title="Styled Button"
        onPress={handlePress}
        buttonStyle={styles.styledButton}
        textStyle={styles.styledText}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f9f9f9',
    padding: 20,
  },
  styledButton: {
    backgroundColor: '#28a745', // Green background for this button
    paddingVertical: 15,
    paddingHorizontal: 30,
  },
  styledText: {
    fontSize: 18,
    color: '#fff',
  },
});

export default App;
```

### 6. Explanation:
- **Default Button**: The first `CustomButton` uses the default styles for both the button and the text.
- **Styled Button**: The second `CustomButton` passes custom styles for the button and the text via the `buttonStyle` and `textStyle` props.
  - The custom button has a green background with extra padding, and the text has a larger font size.

### 7. Styles:
- **buttonStyle**: Allows customization of the button's appearance (background color, padding, etc.).
- **textStyle**: Allows customization of the text inside the button (font size, color, etc.).

### 8. Run the App
Once you've added the `CustomButton` component and updated your `App.js`, run the app on an emulator or a physical device to test it.

For iOS (on macOS):
```bash
npx react-native run-ios
```

For Android:
```bash
npx react-native run-android
```

### 9. Expected Behavior
- You will see two buttons: one with the default styles and another with custom styles (green background and larger text).
- When the user presses either button, an alert will be displayed with the message "Button Pressed!".

---

### Breakdown:
- **Reusable**: The `CustomButton` is reusable with customizable styles and text.
- **Flexibility**: The `buttonStyle` and `textStyle` props make it easy to change the appearance of the button without modifying the component itself.
- **TouchableOpacity**: Provides a user-friendly interactive button that changes opacity when pressed.
