To implement offline functionality in a React Native app with local storage and background sync, you need to store data locally when the user is offline and then sync it with a server once the device is back online. This can be done by leveraging local storage solutions like `AsyncStorage` or `Realm`, and utilizing background tasks for syncing.

Here’s how you can set it up:

### **1. Install Dependencies**

First, you’ll need to install some dependencies for offline storage and background tasks.

- `@react-native-async-storage/async-storage` for local storage.
- `react-native-background-fetch` for background sync tasks.
- `axios` or any other HTTP library for API requests.

```bash
npm install @react-native-async-storage/async-storage axios react-native-background-fetch
```

After installing the dependencies, follow the necessary linking steps for `react-native-background-fetch` based on the platform (Android/iOS).

### **2. Set Up AsyncStorage for Local Data Storage**

We'll use `AsyncStorage` to store data locally while the user is offline. When the app detects the user is back online, it will sync the data with the server.

Create a utility for managing local storage operations, such as storing and retrieving data.

**Example: `localStorage.js`**

```javascript
import AsyncStorage from '@react-native-async-storage/async-storage';

// Store data locally
export const storeDataLocally = async (key, value) => {
  try {
    await AsyncStorage.setItem(key, JSON.stringify(value));
  } catch (e) {
    console.error('Error storing data locally: ', e);
  }
};

// Retrieve stored data
export const getDataLocally = async (key) => {
  try {
    const value = await AsyncStorage.getItem(key);
    return value != null ? JSON.parse(value) : null;
  } catch (e) {
    console.error('Error retrieving data locally: ', e);
    return null;
  }
};

// Remove stored data
export const removeDataLocally = async (key) => {
  try {
    await AsyncStorage.removeItem(key);
  } catch (e) {
    console.error('Error removing data locally: ', e);
  }
};
```

You can use these functions to store data while offline and retrieve it later when the app is back online.

### **3. Implement Background Sync with react-native-background-fetch**

`react-native-background-fetch` allows you to periodically run background tasks, such as syncing local data with the server when the app is not in the foreground.

**Example: Setting up background sync**

In your main app file (e.g., `App.js`), configure the background task to check for network connectivity and sync data if necessary.

```javascript
import React, { useEffect } from 'react';
import { View, Text, Alert } from 'react-native';
import BackgroundFetch from 'react-native-background-fetch';
import { storeDataLocally, getDataLocally } from './localStorage';
import axios from 'axios';

const App = () => {

  useEffect(() => {
    // Configure background fetch
    BackgroundFetch.configure(
      {
        minimumFetchInterval: 15, // In minutes
        stopOnTerminate: false,    // Keep running after app is terminated
        startOnBoot: true,         // Start background fetch after device reboot
        enableHeadless: true,
      },
      async (taskId) => {
        console.log('[BackgroundFetch] task started: ', taskId);

        // Check if there's local data to sync
        const unsyncedData = await getDataLocally('unsyncedData');
        if (unsyncedData) {
          // Sync data with server if network is available
          try {
            const response = await axios.post('https://yourapi.com/sync', unsyncedData);
            if (response.status === 200) {
              console.log('Data synced successfully');
              // Clear local storage after successful sync
              await AsyncStorage.removeItem('unsyncedData');
            }
          } catch (error) {
            console.error('Failed to sync data: ', error);
          }
        }

        // Finish the background task
        BackgroundFetch.finish(taskId);
      },
      (error) => {
        console.error('[BackgroundFetch] failed to start: ', error);
      }
    );

    return () => {
      BackgroundFetch.stop();
    };
  }, []);

  return (
    <View>
      <Text>Offline Functionality with Background Sync</Text>
    </View>
  );
};

export default App;
```

### **4. Store Data Locally When Offline**

Whenever the user performs an action (e.g., adding an item to a shopping cart, submitting a form) while offline, you should store the data locally. If the network is not available, you can store the data in `AsyncStorage` until the app goes back online.

Example of storing unsynced data:

```javascript
import { storeDataLocally } from './localStorage';

// Function to handle form submission or any action
const handleSubmit = async (data) => {
  const isOnline = await checkNetworkStatus(); // Function to check if the user is online
  if (isOnline) {
    // Submit data to the server if online
    try {
      const response = await axios.post('https://yourapi.com/submit', data);
      if (response.status === 200) {
        console.log('Data submitted successfully');
      }
    } catch (error) {
      console.error('Failed to submit data: ', error);
    }
  } else {
    // Store data locally if offline
    await storeDataLocally('unsyncedData', data);
    Alert.alert('You are offline. Data will be synced once you are online.');
  }
};
```

### **5. Detect Network Connectivity**

You can use the `@react-native-community/netinfo` library to check for network status and decide whether to store data locally or submit it to the server.

Install `@react-native-community/netinfo`:

```bash
npm install @react-native-community/netinfo
```

Use it to check the network status:

```javascript
import NetInfo from '@react-native-community/netinfo';

// Function to check if the device is online
const checkNetworkStatus = async () => {
  const state = await NetInfo.fetch();
  return state.isConnected;
};
```

### **6. Handle Network Changes**

You can also subscribe to network status changes and trigger the sync operation when the user goes from offline to online.

```javascript
useEffect(() => {
  const unsubscribe = NetInfo.addEventListener(state => {
    if (state.isConnected) {
      // Trigger sync if internet is back
      BackgroundFetch.start();
    }
  });

  return () => {
    unsubscribe();
  };
}, []);
```

### **7. Optional: Use Realm for More Complex Offline Data Management**

If you need more advanced data management (e.g., handling large datasets or relational data), consider using `Realm` for local storage. Realm provides a robust offline database solution that works seamlessly with React Native.

```bash
npm install realm
```

You can define models and store data offline, and then sync it to the server when the network is available.

### **Conclusion**

By combining `AsyncStorage`, background fetch tasks, and network connectivity detection, you can implement a robust offline functionality in your React Native app. This ensures that users can still interact with the app even when they’re not connected to the internet, and data will be synced automatically once the connection is restored.