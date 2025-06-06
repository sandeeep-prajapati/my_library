### Setting Up Firebase Authentication in React Native

**Objective:**  
Implement Firebase Authentication to enable user sign-up and login in a React Native app.

---

### 1. **Install Firebase in React Native**  

#### Step 1: Install Firebase Dependencies  

Run the following command to install the Firebase library:

```bash
npm install @react-native-firebase/app @react-native-firebase/auth
```

---

### 2. **Set Up Firebase Project**

1. **Create a Firebase Project**  
   - Visit the [Firebase Console](https://console.firebase.google.com/).
   - Click on "Add Project" and follow the setup steps.

2. **Add a Firebase App**  
   - Select your platform (iOS/Android).
   - For iOS, download the `GoogleService-Info.plist` file.  
   - For Android, download the `google-services.json` file.

3. **Integrate Firebase Config Files**
   - **For iOS:** Add `GoogleService-Info.plist` to the Xcode project root.  
   - **For Android:** Place `google-services.json` in the `android/app` directory.

4. **Configure Firebase SDK**
   - **For Android:** Update `android/build.gradle` and `android/app/build.gradle`:
     ```gradle
     // android/build.gradle
     dependencies {
         classpath 'com.google.gms:google-services:4.3.15'
     }

     // android/app/build.gradle
     apply plugin: 'com.google.gms.google-services'
     ```
   - **For iOS:** Run:
     ```bash
     cd ios && pod install
     ```

---

### 3. **Enable Firebase Authentication**  

1. In the Firebase Console, navigate to **Authentication** > **Sign-in method**.  
2. Enable the desired authentication methods (e.g., Email/Password).

---

### 4. **Code Implementation**

#### Step 1: Configure Firebase in the App  

**File:** `firebaseConfig.js`

```javascript
import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';

const firebaseConfig = {
  apiKey: 'YOUR_API_KEY',
  authDomain: 'YOUR_AUTH_DOMAIN',
  projectId: 'YOUR_PROJECT_ID',
  storageBucket: 'YOUR_STORAGE_BUCKET',
  messagingSenderId: 'YOUR_MESSAGING_SENDER_ID',
  appId: 'YOUR_APP_ID',
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
```

---

#### Step 2: Create Signup and Login Components  

**File:** `Signup.js`

```javascript
import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet } from 'react-native';
import { auth } from './firebaseConfig';
import { createUserWithEmailAndPassword } from 'firebase/auth';

const Signup = ({ navigation }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSignup = async () => {
    try {
      await createUserWithEmailAndPassword(auth, email, password);
      alert('Signup successful!');
      navigation.navigate('Login');
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Sign Up</Text>
      {error && <Text style={styles.error}>{error}</Text>}
      <TextInput
        style={styles.input}
        placeholder="Email"
        value={email}
        onChangeText={setEmail}
      />
      <TextInput
        style={styles.input}
        placeholder="Password"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
      />
      <Button title="Sign Up" onPress={handleSignup} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
  },
  header: {
    fontSize: 24,
    marginBottom: 20,
    textAlign: 'center',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 10,
    marginVertical: 10,
    borderRadius: 5,
  },
  error: {
    color: 'red',
    marginBottom: 10,
  },
});

export default Signup;
```

---

**File:** `Login.js`

```javascript
import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet } from 'react-native';
import { auth } from './firebaseConfig';
import { signInWithEmailAndPassword } from 'firebase/auth';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = async () => {
    try {
      await signInWithEmailAndPassword(auth, email, password);
      alert('Login successful!');
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Login</Text>
      {error && <Text style={styles.error}>{error}</Text>}
      <TextInput
        style={styles.input}
        placeholder="Email"
        value={email}
        onChangeText={setEmail}
      />
      <TextInput
        style={styles.input}
        placeholder="Password"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
      />
      <Button title="Login" onPress={handleLogin} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
  },
  header: {
    fontSize: 24,
    marginBottom: 20,
    textAlign: 'center',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 10,
    marginVertical: 10,
    borderRadius: 5,
  },
  error: {
    color: 'red',
    marginBottom: 10,
  },
});

export default Login;
```

---

#### Step 3: Set Up Navigation  

**File:** `App.js`

```javascript
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import Signup from './Signup';
import Login from './Login';

const Stack = createStackNavigator();

const App = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Signup">
        <Stack.Screen name="Signup" component={Signup} />
        <Stack.Screen name="Login" component={Login} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;
```

---

### 5. **Testing Authentication**

1. **Run the App**: Start the development server:
   ```bash
   npx react-native run-android
   # or
   npx react-native run-ios
   ```

2. **Sign Up**: Use the signup form to create a user. Verify in the Firebase Console under **Authentication > Users**.

3. **Log In**: Use the login form with the same credentials to log in.

---

### 6. **Best Practices**

- **Error Handling:** Handle common Firebase errors (e.g., invalid email, weak passwords) with user-friendly messages.
- **Secure Passwords:** Use strong passwords and encourage secure practices.
- **Session Persistence:** Use `onAuthStateChanged` to track user sessions.

---

### 7. **Extensions**

- Add password reset functionality.
- Integrate Google/Facebook sign-in.
- Store user data in Firestore upon successful signup.

This setup ensures a seamless and secure authentication flow for your React Native app!