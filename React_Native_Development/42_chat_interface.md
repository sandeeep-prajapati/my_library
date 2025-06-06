Building a **chat interface with real-time messaging functionality** in React Native requires both the frontend (React Native components) and the backend (real-time communication) setup. Here's how to build a chat app using **React Native**, **Firebase Realtime Database**, and **Firebase Authentication**.

### Steps to Build a Real-Time Chat Interface in React Native

---

### **1. Set up Firebase Project**

First, you need to set up Firebase for your app.

1. **Create a Firebase Project:**
   - Go to [Firebase Console](https://console.firebase.google.com/).
   - Create a new project.
   
2. **Enable Firebase Authentication and Realtime Database:**
   - In the Firebase Console, enable **Authentication** and set up the **Sign-In method** (e.g., Email/Password, Google).
   - Enable **Realtime Database** and set the **rules** to allow read/write access (for development purposes):

     ```json
     {
       "rules": {
         ".read": "auth != null",
         ".write": "auth != null"
       }
     }
     ```

3. **Get Firebase SDK Configuration:**
   - Go to **Project Settings > General > Firebase SDK snippet**.
   - Copy the Firebase config.

---

### **2. Install Dependencies**

Install necessary dependencies for Firebase, React Navigation, and React Native components.

```bash
npm install firebase react-navigation react-navigation-stack react-native-gesture-handler react-native-reanimated
```

Install the Firebase SDK for React Native:

```bash
npm install @react-native-firebase/app @react-native-firebase/auth @react-native-firebase/database
```

For iOS, run the following command to install CocoaPods dependencies:

```bash
cd ios && pod install && cd ..
```

---

### **3. Set up Firebase in Your React Native App**

Configure Firebase in your React Native app by creating a Firebase configuration file (`firebase.js`):

```javascript
// firebase.js
import { initializeApp } from 'firebase/app';
import { getDatabase } from 'firebase/database';
import { getAuth } from 'firebase/auth';

const firebaseConfig = {
  apiKey: 'YOUR_API_KEY',
  authDomain: 'YOUR_AUTH_DOMAIN',
  databaseURL: 'YOUR_DATABASE_URL',
  projectId: 'YOUR_PROJECT_ID',
  storageBucket: 'YOUR_STORAGE_BUCKET',
  messagingSenderId: 'YOUR_SENDER_ID',
  appId: 'YOUR_APP_ID',
};

const app = initializeApp(firebaseConfig);

const auth = getAuth(app);
const database = getDatabase(app);

export { auth, database };
```

---

### **4. Set Up Authentication (Sign-Up & Login)**

For simplicity, we'll use email/password authentication. Create a simple `Login` and `SignUp` screen.

#### **SignUp Screen:**

```javascript
// SignUp.js
import React, { useState } from 'react';
import { View, TextInput, Button, Text } from 'react-native';
import { auth } from './firebase';
import { createUserWithEmailAndPassword } from 'firebase/auth';

const SignUp = ({ navigation }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSignUp = async () => {
    try {
      await createUserWithEmailAndPassword(auth, email, password);
      navigation.navigate('Chat');
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <View>
      <TextInput
        placeholder="Email"
        value={email}
        onChangeText={setEmail}
      />
      <TextInput
        placeholder="Password"
        secureTextEntry
        value={password}
        onChangeText={setPassword}
      />
      {error && <Text>{error}</Text>}
      <Button title="Sign Up" onPress={handleSignUp} />
    </View>
  );
};

export default SignUp;
```

#### **Login Screen:**

```javascript
// Login.js
import React, { useState } from 'react';
import { View, TextInput, Button, Text } from 'react-native';
import { auth } from './firebase';
import { signInWithEmailAndPassword } from 'firebase/auth';

const Login = ({ navigation }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = async () => {
    try {
      await signInWithEmailAndPassword(auth, email, password);
      navigation.navigate('Chat');
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <View>
      <TextInput
        placeholder="Email"
        value={email}
        onChangeText={setEmail}
      />
      <TextInput
        placeholder="Password"
        secureTextEntry
        value={password}
        onChangeText={setPassword}
      />
      {error && <Text>{error}</Text>}
      <Button title="Login" onPress={handleLogin} />
    </View>
  );
};

export default Login;
```

---

### **5. Set Up Chat Screen**

To build a real-time chat, use Firebase's Realtime Database to send and receive messages.

#### **Chat Screen:**

```javascript
// Chat.js
import React, { useState, useEffect } from 'react';
import { View, TextInput, Button, FlatList, Text } from 'react-native';
import { database, auth } from './firebase';
import { ref, push, onValue } from 'firebase/database';

const Chat = () => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    const messagesRef = ref(database, 'messages/');
    onValue(messagesRef, (snapshot) => {
      const data = snapshot.val();
      const messageList = data ? Object.values(data) : [];
      setMessages(messageList);
    });
  }, []);

  const sendMessage = () => {
    if (message.trim() === '') return;
    const messagesRef = ref(database, 'messages/');
    push(messagesRef, {
      text: message,
      user: auth.currentUser.email,
      timestamp: Date.now(),
    });
    setMessage('');
  };

  return (
    <View style={{ flex: 1, padding: 10 }}>
      <FlatList
        data={messages}
        keyExtractor={(item, index) => index.toString()}
        renderItem={({ item }) => (
          <View>
            <Text>{item.user}: {item.text}</Text>
          </View>
        )}
      />
      <TextInput
        value={message}
        onChangeText={setMessage}
        placeholder="Type a message"
      />
      <Button title="Send" onPress={sendMessage} />
    </View>
  );
};

export default Chat;
```

---

### **6. Set Up Navigation**

For navigation between screens (Login, SignUp, and Chat), use React Navigation:

1. **Install React Navigation:**

```bash
npm install @react-navigation/native @react-navigation/stack react-native-gesture-handler react-native-reanimated
```

2. **Set Up Navigation:**

```javascript
// App.js
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import SignUp from './SignUp';
import Login from './Login';
import Chat from './Chat';

const Stack = createStackNavigator();

const App = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Login">
        <Stack.Screen name="SignUp" component={SignUp} />
        <Stack.Screen name="Login" component={Login} />
        <Stack.Screen name="Chat" component={Chat} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;
```

---

### **7. Final Touches and Running the App**

1. **Run the App**:
   - Ensure your Firebase Realtime Database and Authentication are set up correctly.
   - Start the app on a simulator/emulator or a physical device.

2. **Testing**:
   - Test by creating multiple accounts, logging in, and sending/receiving messages in real-time.
   - Firebase Realtime Database will automatically sync messages between all devices in real-time.

---

### **Conclusion**

By following these steps, you've built a **real-time chat app** using **Firebase** for user authentication and messaging. The app supports login, sign-up, and real-time messaging. You can expand this by adding features like notifications, message timestamps, image uploads, and more.