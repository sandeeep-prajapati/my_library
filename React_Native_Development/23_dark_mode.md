

---

## **Steps to Implement Dark Mode**

### 1. **Setup Theme Context**
- Use the Context API to manage the theme state across the app.

---

### 2. **Create Light and Dark Themes**
- Define light and dark color schemes.

---

### 3. **Add a Toggle Button**
- Provide a UI toggle to switch between themes.

---

### 4. **Example Implementation**

#### **Full Code**

```jsx
import React, { createContext, useContext, useState } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';

// Create a Theme Context
const ThemeContext = createContext();

const App = () => {
  // State to manage theme (light/dark)
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Toggle function
  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  // Define light and dark themes
  const theme = isDarkMode
    ? {
        backgroundColor: '#121212',
        textColor: '#ffffff',
        buttonColor: '#bb86fc',
      }
    : {
        backgroundColor: '#ffffff',
        textColor: '#000000',
        buttonColor: '#6200ee',
      };

  return (
    // Provide the theme to the entire app
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      <ThemedApp />
    </ThemeContext.Provider>
  );
};

// Themed App Component
const ThemedApp = () => {
  const { theme, toggleTheme } = useContext(ThemeContext);

  return (
    <View style={[styles.container, { backgroundColor: theme.backgroundColor }]}>
      <Text style={[styles.text, { color: theme.textColor }]}>
        {theme.backgroundColor === '#121212' ? 'Dark Mode' : 'Light Mode'}
      </Text>
      <Button
        title="Toggle Theme"
        color={theme.buttonColor}
        onPress={toggleTheme}
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
    fontSize: 20,
    marginBottom: 20,
  },
});

export default App;
```

---

### **Explanation**

1. **Theme Context**
   - `ThemeContext` is created to manage and provide the theme across the app using `createContext`.

2. **Theme State**
   - `isDarkMode` is a boolean state to toggle between light and dark modes.

3. **Themes**
   - Define separate color schemes for light and dark modes.

4. **Toggle Function**
   - The `toggleTheme` function switches the `isDarkMode` state.

5. **Dynamic Styling**
   - Use the theme object to dynamically style components:
     ```javascript
     backgroundColor: theme.backgroundColor
     color: theme.textColor
     ```

---

### **Customizing the Button**
If you want a custom switch instead of a button, use `Switch` from React Native:
```jsx
import { Switch } from 'react-native';

// Replace the Button with:
<Switch
  value={isDarkMode}
  onValueChange={toggleTheme}
  trackColor={{ false: '#767577', true: '#81b0ff' }}
  thumbColor={isDarkMode ? '#f5dd4b' : '#f4f3f4'}
/>
```

---

### **React Native Appearance API (Optional)**
To automatically detect the system theme:
1. Import the `useColorScheme` hook:
   ```javascript
   import { useColorScheme } from 'react-native';
   ```
2. Use it to determine the default theme:
   ```javascript
   const systemTheme = useColorScheme(); // 'light' or 'dark'
   const [isDarkMode, setIsDarkMode] = useState(systemTheme === 'dark');
   ```

---

### **Extending Functionality**
- **Persist Theme:** Save the selected theme using AsyncStorage or SecureStore to maintain it across app launches.
- **Theming Libraries:** Use libraries like `react-native-paper` or `styled-components` for advanced theming capabilities.
