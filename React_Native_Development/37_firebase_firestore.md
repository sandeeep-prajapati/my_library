### Storing and Retrieving App Data Using Firebase Firestore in React Native

**Objective:**  
Learn how to store and retrieve data in Firebase Firestore, a NoSQL cloud database, for a React Native app.

---

### 1. **Install Firestore Dependencies**

Run the following command to add Firebase Firestore to your project:

```bash
npm install @react-native-firebase/firestore
```

---

### 2. **Set Up Firestore in Firebase Console**

1. **Enable Firestore**  
   - Go to the Firebase Console.
   - Select your project.
   - Navigate to **Firestore Database** > **Create Database**.
   - Choose a **Start in Test Mode** or **Production Mode** and click **Next**.

2. **Set Up Firestore Rules**  
   Test Mode allows open access. For production, configure rules to secure your database:
   ```json
   rules_version = '2';
   service cloud.firestore {
     match /databases/{database}/documents {
       match /{document=**} {
         allow read, write: if request.auth != null;
       }
     }
   }
   ```

---

### 3. **Firestore Configuration in App**

**File:** `firebaseConfig.js`

```javascript
import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: 'YOUR_API_KEY',
  authDomain: 'YOUR_AUTH_DOMAIN',
  projectId: 'YOUR_PROJECT_ID',
  storageBucket: 'YOUR_STORAGE_BUCKET',
  messagingSenderId: 'YOUR_MESSAGING_SENDER_ID',
  appId: 'YOUR_APP_ID',
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
```

---

### 4. **Store Data in Firestore**

#### Example: Adding a New User

**File:** `AddUser.js`

```javascript
import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet } from 'react-native';
import { collection, addDoc } from 'firebase/firestore';
import { db } from './firebaseConfig';

const AddUser = () => {
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [message, setMessage] = useState('');

  const handleAddUser = async () => {
    try {
      const docRef = await addDoc(collection(db, 'users'), {
        name,
        age: parseInt(age, 10),
      });
      setMessage(`User added with ID: ${docRef.id}`);
    } catch (error) {
      setMessage(`Error adding user: ${error.message}`);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Add User</Text>
      {message && <Text style={styles.message}>{message}</Text>}
      <TextInput
        style={styles.input}
        placeholder="Name"
        value={name}
        onChangeText={setName}
      />
      <TextInput
        style={styles.input}
        placeholder="Age"
        value={age}
        onChangeText={setAge}
        keyboardType="numeric"
      />
      <Button title="Add User" onPress={handleAddUser} />
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
  message: {
    color: 'green',
    marginBottom: 10,
    textAlign: 'center',
  },
});

export default AddUser;
```

---

### 5. **Retrieve Data from Firestore**

#### Example: Fetch and Display Users

**File:** `FetchUsers.js`

```javascript
import React, { useState, useEffect } from 'react';
import { View, Text, FlatList, StyleSheet } from 'react-native';
import { collection, getDocs } from 'firebase/firestore';
import { db } from './firebaseConfig';

const FetchUsers = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const querySnapshot = await getDocs(collection(db, 'users'));
        const usersList = querySnapshot.docs.map(doc => ({
          id: doc.id,
          ...doc.data(),
        }));
        setUsers(usersList);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching users: ', error);
        setLoading(false);
      }
    };

    fetchUsers();
  }, []);

  if (loading) {
    return <Text style={styles.loading}>Loading...</Text>;
  }

  return (
    <View style={styles.container}>
      <Text style={styles.header}>User List</Text>
      <FlatList
        data={users}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <View style={styles.item}>
            <Text style={styles.name}>{item.name}</Text>
            <Text>Age: {item.age}</Text>
          </View>
        )}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  header: {
    fontSize: 24,
    marginBottom: 20,
    textAlign: 'center',
  },
  loading: {
    flex: 1,
    textAlign: 'center',
    marginTop: 50,
  },
  item: {
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  name: {
    fontSize: 18,
    fontWeight: 'bold',
  },
});

export default FetchUsers;
```

---

### 6. **Run the App**

1. Start the development server:
   ```bash
   npx react-native run-android
   # or
   npx react-native run-ios
   ```

2. Use `AddUser` to add user data to Firestore.  
3. Use `FetchUsers` to display the list of users retrieved from Firestore.

---

### 7. **Best Practices**

- **Data Validation:** Validate data before storing it in Firestore to ensure consistency.
- **Pagination:** For large datasets, implement Firestore’s `startAfter` or `limit` for pagination.
- **Security Rules:** Configure Firestore rules to restrict read/write access as per your app's requirements.
- **Error Handling:** Always handle errors gracefully in both storing and fetching operations.

---

### 8. **Extensions**

- Add real-time updates with Firestore’s `onSnapshot`.
- Use Firestore queries to filter, sort, or search data.
- Integrate Firestore with Firebase Authentication to store user-specific data. 

This setup enables you to efficiently store and retrieve app data using Firebase Firestore in React Native.