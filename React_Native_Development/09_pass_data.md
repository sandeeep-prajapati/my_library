To pass data between two screens using **navigation params** in React Native with React Navigation, you can use the `navigation.navigate` function to pass parameters from one screen to another, and retrieve those parameters on the destination screen using `route.params`.

Here’s how you can create a flow where data is passed between two screens using **navigation params**:

### 1. **Install React Navigation Libraries**

First, ensure that you have the necessary libraries installed. If you haven't done so yet, follow these steps to install the required dependencies for React Navigation:

```bash
npm install @react-navigation/native
npm install @react-navigation/stack
npm install react-native-screens react-native-safe-area-context
```

### 2. **Create the Screens**

We’ll create two screens: `HomeScreen` and `DetailScreen`. Data (for example, a name) will be passed from `HomeScreen` to `DetailScreen`.

#### **HomeScreen.js**
This screen will allow the user to input a name and pass it as a parameter to `DetailScreen`.

```javascript
import React, { useState } from 'react';
import { View, TextInput, Button, StyleSheet } from 'react-native';

const HomeScreen = ({ navigation }) => {
  const [name, setName] = useState('');

  const handleNavigate = () => {
    navigation.navigate('Detail', {
      name: name, // passing the name parameter
    });
  };

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        placeholder="Enter your name"
        value={name}
        onChangeText={setName}
      />
      <Button title="Go to Detail Screen" onPress={handleNavigate} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 16,
  },
  input: {
    width: '80%',
    padding: 10,
    marginBottom: 20,
    borderWidth: 1,
    borderRadius: 5,
  },
});

export default HomeScreen;
```

#### **DetailScreen.js**
This screen will retrieve the `name` parameter passed from `HomeScreen` and display it.

```javascript
import React from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';

const DetailScreen = ({ route, navigation }) => {
  // Accessing the passed parameter from the navigation
  const { name } = route.params;

  return (
    <View style={styles.container}>
      <Text style={styles.text}>Hello, {name}!</Text>
      <Button
        title="Go Back to Home Screen"
        onPress={() => navigation.goBack()}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 16,
  },
  text: {
    fontSize: 24,
    marginBottom: 20,
  },
});

export default DetailScreen;
```

### 3. **Set Up Navigation**

In the `App.js` file, set up the navigation using a Stack Navigator.

#### **App.js**
```javascript
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import HomeScreen from './HomeScreen';
import DetailScreen from './DetailScreen';

const Stack = createStackNavigator();

const App = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="Detail" component={DetailScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;
```

### 4. **How It Works:**

- **HomeScreen**: 
  - The user enters a name in the `TextInput` field. The `name` is stored in the state using `useState`.
  - When the "Go to Detail Screen" button is pressed, the `handleNavigate` function is triggered. This function uses `navigation.navigate('Detail', { name: name })` to pass the `name` to the `DetailScreen`.
  
- **DetailScreen**:
  - The `DetailScreen` accesses the `name` parameter through `route.params.name`.
  - The `name` is displayed in a `Text` component.
  - There's also a "Go Back to Home Screen" button that takes the user back to the `HomeScreen` using `navigation.goBack()`.

### 5. **Running the App**

Now, you can run the app and test the flow between the screens:

For **iOS** (on macOS):
```bash
npx react-native run-ios
```

For **Android**:
```bash
npx react-native run-android
```

### 6. **Explanation:**

- **Passing Params**: In `HomeScreen`, when you call `navigation.navigate('Detail', { name: name })`, you're passing the `name` parameter to `DetailScreen` via navigation params.
- **Retrieving Params**: In `DetailScreen`, the `name` is accessed using `route.params.name`. `route.params` contains all the parameters passed when navigating to that screen.
- **Navigating Back**: In `DetailScreen`, `navigation.goBack()` is used to go back to the previous screen.

### 7. **Customizing and Extending**:
- You can pass more parameters and use them in your screens by expanding the `navigation.navigate` and `route.params` objects.
- You can also use `navigation.setParams` to update the parameters dynamically if required.

---

This implementation demonstrates how you can pass data between screens using React Navigation and navigation params, allowing you to create dynamic, data-driven applications in React Native.