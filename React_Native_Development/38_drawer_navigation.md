### Building a Custom Drawer Navigation Menu with React Navigation

**Objective:**  
Create a custom drawer navigation menu in a React Native app using React Navigation to enhance the user experience.

---

### 1. **Install Dependencies**

Install the required packages for React Navigation and Drawer Navigation:

```bash
npm install @react-navigation/native @react-navigation/drawer react-native-screens react-native-safe-area-context react-native-gesture-handler react-native-reanimated react-native-vector-icons
```

**Note:** If you’re using Expo, run:

```bash
expo install react-native-gesture-handler react-native-reanimated react-native-screens react-native-safe-area-context react-native-vector-icons @react-navigation/native @react-navigation/drawer
```

---

### 2. **Set Up React Navigation**

Wrap your app in the Navigation Container:

**File:** `App.js`

```javascript
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import DrawerNavigator from './DrawerNavigator';

export default function App() {
  return (
    <NavigationContainer>
      <DrawerNavigator />
    </NavigationContainer>
  );
}
```

---

### 3. **Install Drawer Navigator**

Create a new file for the drawer navigation logic.

**File:** `DrawerNavigator.js`

```javascript
import React from 'react';
import { createDrawerNavigator } from '@react-navigation/drawer';
import HomeScreen from './screens/HomeScreen';
import ProfileScreen from './screens/ProfileScreen';
import SettingsScreen from './screens/SettingsScreen';
import CustomDrawerContent from './components/CustomDrawerContent';

const Drawer = createDrawerNavigator();

const DrawerNavigator = () => {
  return (
    <Drawer.Navigator
      drawerContent={(props) => <CustomDrawerContent {...props} />}
      screenOptions={{
        headerShown: true,
        drawerStyle: { backgroundColor: '#f5f5f5', width: 250 },
      }}
    >
      <Drawer.Screen name="Home" component={HomeScreen} />
      <Drawer.Screen name="Profile" component={ProfileScreen} />
      <Drawer.Screen name="Settings" component={SettingsScreen} />
    </Drawer.Navigator>
  );
};

export default DrawerNavigator;
```

---

### 4. **Create Screens**

Create basic screens to navigate.

**File:** `screens/HomeScreen.js`

```javascript
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const HomeScreen = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>Welcome to the Home Screen</Text>
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
  },
});

export default HomeScreen;
```

**File:** `screens/ProfileScreen.js`

```javascript
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const ProfileScreen = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>Welcome to the Profile Screen</Text>
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
  },
});

export default ProfileScreen;
```

**File:** `screens/SettingsScreen.js`

```javascript
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const SettingsScreen = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>Welcome to the Settings Screen</Text>
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
  },
});

export default SettingsScreen;
```

---

### 5. **Customize the Drawer Content**

Create a custom drawer component.

**File:** `components/CustomDrawerContent.js`

```javascript
import React from 'react';
import {
  DrawerContentScrollView,
  DrawerItemList,
} from '@react-navigation/drawer';
import { View, Text, StyleSheet, Image, TouchableOpacity } from 'react-native';

const CustomDrawerContent = (props) => {
  return (
    <DrawerContentScrollView {...props}>
      {/* Header Section */}
      <View style={styles.header}>
        <Image
          source={{ uri: 'https://via.placeholder.com/100' }}
          style={styles.profileImage}
        />
        <Text style={styles.username}>John Doe</Text>
      </View>

      {/* Drawer Items */}
      <DrawerItemList {...props} />

      {/* Custom Footer */}
      <View style={styles.footer}>
        <TouchableOpacity
          style={styles.logoutButton}
          onPress={() => console.log('Logout Pressed')}
        >
          <Text style={styles.logoutText}>Logout</Text>
        </TouchableOpacity>
      </View>
    </DrawerContentScrollView>
  );
};

const styles = StyleSheet.create({
  header: {
    padding: 20,
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  profileImage: {
    width: 80,
    height: 80,
    borderRadius: 40,
    marginBottom: 10,
  },
  username: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  footer: {
    marginTop: 20,
    paddingHorizontal: 10,
  },
  logoutButton: {
    backgroundColor: '#ff5c5c',
    paddingVertical: 10,
    borderRadius: 5,
    alignItems: 'center',
  },
  logoutText: {
    color: '#fff',
    fontWeight: 'bold',
  },
});

export default CustomDrawerContent;
```

---

### 6. **Run the App**

1. Start the development server:
   ```bash
   npx react-native run-android
   # or
   npx react-native run-ios
   ```

2. Open the drawer menu by swiping from the left edge or by pressing the menu icon.

---

### 7. **Features of the Custom Drawer**

- **Header Section**: Displays a profile image and username.  
- **Drawer Items**: Automatically maps the screens in the navigation stack.  
- **Footer Section**: Includes a logout button with custom logic.

---

### 8. **Extend Functionality**

- Add icons to drawer items using `react-native-vector-icons`.
- Use `useNavigation` for navigation actions inside the custom drawer.
- Fetch user details dynamically and display them in the header.

This implementation provides a visually appealing and functional custom drawer navigation menu for your React Native app.