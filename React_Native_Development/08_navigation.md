To implement navigation between two screens in React Native, we can use **React Navigation**, which is a popular library for navigating between different screens in a React Native app.

### Steps to Implement Navigation Between Two Screens:

1. **Install React Navigation and Dependencies**
2. **Create Two Screens**
3. **Setup Navigation**
4. **Test Navigation between Screens**

### 1. **Install React Navigation and Dependencies**

First, you need to install the required libraries for React Navigation.

- Install the core package of React Navigation:
```bash
npm install @react-navigation/native
```

- Install the required dependencies:
```bash
npm install react-native-screens react-native-safe-area-context
```

- Install the stack navigator for screen-to-screen navigation:
```bash
npm install @react-navigation/stack
```

- If you're using Expo, these steps are unnecessary, as Expo comes with these libraries pre-installed. For non-Expo projects, run:
```bash
npx react-native link react-native-screens react-native-safe-area-context
```

### 2. **Create Two Screens**

In this example, we’ll create two screens: `HomeScreen` and `DetailScreen`.

#### **HomeScreen.js**
```javascript
import React from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';

const HomeScreen = ({ navigation }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>Welcome to the Home Screen!</Text>
      <Button
        title="Go to Detail Screen"
        onPress={() => navigation.navigate('Detail')}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  text: {
    fontSize: 18,
    marginBottom: 20,
  },
});

export default HomeScreen;
```

#### **DetailScreen.js**
```javascript
import React from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';

const DetailScreen = ({ navigation }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>This is the Detail Screen!</Text>
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
  },
  text: {
    fontSize: 18,
    marginBottom: 20,
  },
});

export default DetailScreen;
```

### 3. **Setup Navigation**

Now, set up React Navigation in your main app file (`App.js` or `App.tsx`).

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

### 4. **Explanation:**

- **`NavigationContainer`**: This component manages the navigation tree and must wrap your app's entire navigation structure.
- **`createStackNavigator`**: Creates a stack navigator that allows users to navigate through different screens in a stack-based manner (e.g., push, pop).
- **`Stack.Navigator`**: This component manages the navigation stack. We define the `initialRouteName` as "Home", which means the app starts at the `HomeScreen`.
- **`Stack.Screen`**: Each screen is added to the navigator using this component. The `name` is the route name, and the `component` is the screen to be displayed when that route is accessed.

### 5. **Running the App**

Now, you can run the app to test the navigation:

For iOS (on macOS):
```bash
npx react-native run-ios
```

For Android:
```bash
npx react-native run-android
```

### 6. **How It Works:**

- On the **Home Screen**, there's a button labeled "Go to Detail Screen." When pressed, it triggers the `navigation.navigate('Detail')` method, which navigates to the `DetailScreen`.
- On the **Detail Screen**, there's a button labeled "Go Back to Home Screen." When pressed, it calls `navigation.goBack()`, which takes the user back to the previous screen (in this case, the `HomeScreen`).

### 7. **Customization and Improvements**:
- You can pass parameters between screens using `navigation.navigate('Detail', { param: 'value' })` and retrieve them using `route.params`.
- You can customize the header bar, animations, and transitions between screens using the options in `Stack.Navigator` and `Stack.Screen`.

---

### Conclusion:
You have successfully set up navigation between two screens in React Native using React Navigation. The `HomeScreen` and `DetailScreen` are connected using a Stack Navigator, and the app demonstrates how to navigate from one screen to another.