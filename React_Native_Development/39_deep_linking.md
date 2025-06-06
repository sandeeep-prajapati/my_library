### Enabling Deep Linking in a React Native App

**Objective:**  
Set up deep linking in a React Native app to allow users to navigate directly to specific screens via URLs.

---

### 1. **Install Dependencies**

Ensure you have React Navigation installed. If not, install it along with the required dependencies:

```bash
npm install @react-navigation/native react-native-screens react-native-safe-area-context react-native-gesture-handler react-native-reanimated react-native-vector-icons
```

---

### 2. **Configure Deep Linking in the Navigation Container**

Modify your navigation setup to support deep linking.

#### Example Navigation Setup:

**File:** `App.js`

```javascript
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';

const Stack = createStackNavigator();

import HomeScreen from './screens/HomeScreen';
import ProfileScreen from './screens/ProfileScreen';
import SettingsScreen from './screens/SettingsScreen';

export default function App() {
  const linking = {
    prefixes: ['myapp://', 'https://myapp.com'],
    config: {
      screens: {
        Home: '',
        Profile: 'profile/:userId',
        Settings: 'settings',
      },
    },
  };

  return (
    <NavigationContainer linking={linking}>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="Profile" component={ProfileScreen} />
        <Stack.Screen name="Settings" component={SettingsScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
```

---

### 3. **Set Up Screens**

#### `HomeScreen.js`
```javascript
import React from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';

const HomeScreen = ({ navigation }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>Home Screen</Text>
      <Button
        title="Go to Profile"
        onPress={() => navigation.navigate('Profile', { userId: 123 })}
      />
      <Button
        title="Go to Settings"
        onPress={() => navigation.navigate('Settings')}
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
  },
});

export default HomeScreen;
```

#### `ProfileScreen.js`
```javascript
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const ProfileScreen = ({ route }) => {
  const { userId } = route.params || {};

  return (
    <View style={styles.container}>
      <Text style={styles.text}>Profile Screen</Text>
      <Text>User ID: {userId}</Text>
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

#### `SettingsScreen.js`
```javascript
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const SettingsScreen = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>Settings Screen</Text>
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

### 4. **Test Deep Linking**

#### Using `npx uri-scheme` (Dev Environment)
1. Install `uri-scheme` globally:
   ```bash
   npm install -g uri-scheme
   ```
2. Test deep links:
   ```bash
   uri-scheme open myapp://profile/123 --android
   uri-scheme open myapp://settings --android
   ```

#### Using Expo (If Applicable)
- Add deep linking configuration in `app.json` or `app.config.js`:
  ```json
  {
    "expo": {
      "scheme": "myapp"
    }
  }
  ```
- Test with:
  ```bash
  expo start
  ```

#### Using a Browser for Web Support
Visit:  
`https://myapp.com/profile/123`  
`https://myapp.com/settings`

---

### 5. **Configure Platforms**

#### **Android Setup**
1. Open `android/app/src/main/AndroidManifest.xml` and add:
   ```xml
   <intent-filter>
       <action android:name="android.intent.action.VIEW" />
       <category android:name="android.intent.category.DEFAULT" />
       <category android:name="android.intent.category.BROWSABLE" />
       <data android:scheme="myapp" android:host="myapp.com" />
   </intent-filter>
   ```

#### **iOS Setup**
1. Open `ios/YourApp/AppDelegate.m` and add:
   ```objective-c
   - (BOOL)application:(UIApplication *)application 
               openURL:(NSURL *)url 
               options:(NSDictionary<UIApplicationOpenURLOptionsKey, id> *)options {
       return [RCTLinkingManager application:application openURL:url options:options];
   }
   ```
2. Add URL Types in `Info.plist`:
   ```xml
   <key>CFBundleURLTypes</key>
   <array>
       <dict>
           <key>CFBundleURLSchemes</key>
           <array>
               <string>myapp</string>
           </array>
       </dict>
   </array>
   ```

---

### 6. **Handle Navigation and Params**

When navigating via a deep link, React Navigation automatically parses the URL and passes the params to the appropriate screen.  

For example:  
- **URL:** `myapp://profile/123`  
- **Param on ProfileScreen:** `userId = 123`

---

### 7. **Debugging**
- Use `console.log(route.params)` in screens to verify parameters.
- Test on both real devices and emulators for proper behavior.

---

With this setup, users can seamlessly navigate to specific screens in your React Native app via deep links!