To build a simple form with text input fields and a submit button in React Native, follow these steps:

### 1. Set Up the Project
If you haven't already set up a React Native project, create one by running:

```bash
npx react-native init SimpleFormApp
```

Navigate to the project directory:

```bash
cd SimpleFormApp
```

### 2. Modify the `App.js` File
Open the `App.js` file and replace its content with the following code to create a form with text input fields and a submit button:

```javascript
import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet, Alert } from 'react-native';

const App = () => {
  // State variables to store input values
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');

  // Handle form submission
  const handleSubmit = () => {
    // Check if both fields are filled
    if (!name || !email) {
      Alert.alert('Error', 'Please fill in both fields');
      return;
    }

    // Display the form data in an alert
    Alert.alert('Form Submitted', `Name: ${name}\nEmail: ${email}`);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.heading}>Simple Form</Text>

      <TextInput
        style={styles.input}
        placeholder="Enter your name"
        value={name}
        onChangeText={setName}
      />

      <TextInput
        style={styles.input}
        placeholder="Enter your email"
        value={email}
        onChangeText={setEmail}
        keyboardType="email-address"
      />

      <Button title="Submit" onPress={handleSubmit} />
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
  heading: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  input: {
    height: 40,
    borderColor: '#ccc',
    borderWidth: 1,
    borderRadius: 4,
    width: '100%',
    marginBottom: 20,
    paddingHorizontal: 10,
  },
});

export default App;
```

### 3. Explanation of the Code:
- **State Variables**: `name` and `email` are state variables that hold the values of the text input fields.
  - `useState('')` is used to initialize the state for `name` and `email` as empty strings.
- **`TextInput` Components**: These are the text input fields for the form. The `value` prop binds the input field to the state variable, and the `onChangeText` prop updates the state as the user types.
- **Submit Button**: The `Button` component triggers the `handleSubmit` function when pressed.
- **Form Validation**: In the `handleSubmit` function, it checks if both fields are filled. If not, an alert is shown to prompt the user to fill in both fields.
- **Alert**: Once the form is successfully submitted, the entered data is shown in an alert.

### 4. Run the App
After modifying the `App.js` file, run the app on an emulator or a physical device.

For iOS (on macOS):
```bash
npx react-native run-ios
```

For Android:
```bash
npx react-native run-android
```

### 5. Interaction
- **Text Inputs**: You can enter your name and email in the text input fields.
- **Submit Button**: When you press the "Submit" button, if both fields are filled, it will display the form data in an alert. If any field is empty, it will prompt the user to fill both fields.

---

### Breakdown:
- **State Management**: React state (`useState`) is used to store and manage input field values.
- **Text Input Fields**: React Native's `TextInput` component is used for accepting user input.
- **Button**: The `Button` component triggers the form submission.
- **Form Validation**: Simple validation checks are added to ensure both fields are filled before submitting.

This example demonstrates how to create a simple form in React Native with text input fields and a submit button. You can expand it by adding more fields, custom styles, or connecting it to a backend API for data submission.