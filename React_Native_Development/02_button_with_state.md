To create a button that toggles a message on and off when clicked using React state, follow these steps:

### 1. Set Up the Project
If you haven’t already set up a React Native project, create one by running the following command:

```bash
npx react-native init ToggleMessageApp
```

Navigate to the project directory:

```bash
cd ToggleMessageApp
```

### 2. Modify the `App.js` File
Open the `App.js` file and replace its content with the following code:

```javascript
import React, { useState } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';

const App = () => {
  // Set up state to track the visibility of the message
  const [isVisible, setIsVisible] = useState(false);

  // Toggle the visibility when the button is pressed
  const toggleMessage = () => {
    setIsVisible(!isVisible);
  };

  return (
    <View style={styles.container}>
      <Button title="Toggle Message" onPress={toggleMessage} />
      {isVisible && <Text style={styles.message}>Hello, World!</Text>}
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
  message: {
    marginTop: 20,
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
});

export default App;
```

### 3. Explanation of the Code:
- **`useState(false)`**: We use React's `useState` hook to manage the state of the message's visibility. Initially, the message is not visible (`false`).
- **`toggleMessage` function**: This function toggles the visibility state between `true` and `false` whenever the button is pressed.
- **`Button` component**: This is a React Native component that renders a clickable button. When the button is pressed, it triggers the `toggleMessage` function.
- **Conditional rendering**: The `{isVisible && <Text>}` syntax ensures that the message is only displayed when `isVisible` is `true`.

### 4. Run the App
After modifying the code, run the app on an emulator or a physical device.

For iOS (on macOS):
```bash
npx react-native run-ios
```

For Android:
```bash
npx react-native run-android
```

### 5. Interaction
- When you press the "Toggle Message" button, the message "Hello, World!" will appear.
- Pressing the button again will hide the message.

---

### Breakdown:
- **State Management**: We use React state (`useState`) to control whether the message is visible or hidden.
- **Button**: A simple `Button` component triggers the state change.
- **Conditional Rendering**: The message is rendered only when `isVisible` is `true`.

This example demonstrates a basic but powerful pattern in React Native — managing UI components based on state.