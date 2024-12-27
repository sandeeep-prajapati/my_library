### Notes for Setting Up Push Notifications with Firebase Cloud Messaging (FCM) in Expo

#### Prerequisites
1. Basic knowledge of React Native and Expo.
2. An Expo project set up (managed workflow).
3. A Firebase project created in the Firebase Console (https://console.firebase.google.com).

---

### Steps to Implement Push Notifications in Expo

#### 1. **Install Required Libraries**
Expo provides a built-in library for handling notifications.

Run the following command to install the notifications library:
```bash
expo install expo-notifications
```

---

#### 2. **Set Up Firebase Project**

##### a. **Create a Firebase Project**
1. Go to [Firebase Console](https://console.firebase.google.com/).
2. Click **Add Project**, follow the steps, and register your app.

##### b. **Download Configuration Files**
- Download the `google-services.json` file for Android.
- Download the `GoogleService-Info.plist` file for iOS.
  
For managed Expo projects, these files will not be directly used, but keep them ready if you transition to a bare workflow.

##### c. **Set Up Firebase Cloud Messaging**
1. Go to **Project Settings** in Firebase.
2. Under the **Cloud Messaging** tab, enable Firebase Cloud Messaging (FCM).
3. Note the **Server Key** and **Sender ID** for later use.

---

#### 3. **Request Notification Permissions**
Ask the user for permission to send notifications.

**Code Example:**
```javascript
import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import { Alert } from 'react-native';

export const registerForPushNotifications = async () => {
  if (Device.isDevice) {
    const { status: existingStatus } = await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;

    if (existingStatus !== 'granted') {
      const { status } = await Notifications.requestPermissionsAsync();
      finalStatus = status;
    }

    if (finalStatus !== 'granted') {
      Alert.alert('Failed to get push token for notifications!');
      return null;
    }

    const token = (await Notifications.getExpoPushTokenAsync()).data;
    console.log('Expo Push Token:', token);
    return token;
  } else {
    Alert.alert('Must use physical device for Push Notifications');
  }
};
```

---

#### 4. **Configure Notification Behavior**
Expo allows customization of how notifications behave when received.

**Foreground Notifications Handling:**
```javascript
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: false,
    shouldSetBadge: false,
  }),
});
```

---

#### 5. **Send a Push Notification**
- Use the Expo Push Notification Tool (https://expo.dev/notifications) to send test notifications.
- You’ll need the Expo Push Token generated in the earlier steps.

**Send via REST API:**
You can send notifications programmatically using `fetch`.

```javascript
const sendPushNotification = async (expoPushToken) => {
  const message = {
    to: expoPushToken,
    sound: 'default',
    title: 'Test Notification',
    body: 'This is a test message!',
    data: { extraData: 'Some extra data' },
  };

  await fetch('https://exp.host/--/api/v2/push/send', {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(message),
  });
};
```

---

#### 6. **Example App**
Here’s a complete example integrating the steps above:

```javascript
import React, { useEffect } from 'react';
import { View, Text, Button, StyleSheet, Alert } from 'react-native';
import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';

const App = () => {
  useEffect(() => {
    registerForPushNotifications();

    const subscription = Notifications.addNotificationReceivedListener((notification) => {
      Alert.alert('Notification Received', notification.request.content.body);
    });

    return () => subscription.remove();
  }, []);

  const sendNotification = async () => {
    const token = await registerForPushNotifications();
    if (token) {
      sendPushNotification(token);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Expo Push Notifications</Text>
      <Button title="Send Notification" onPress={sendNotification} />
    </View>
  );
};

const registerForPushNotifications = async () => {
  if (Device.isDevice) {
    const { status: existingStatus } = await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;

    if (existingStatus !== 'granted') {
      const { status } = await Notifications.requestPermissionsAsync();
      finalStatus = status;
    }

    if (finalStatus !== 'granted') {
      Alert.alert('Failed to get push token for notifications!');
      return null;
    }

    const token = (await Notifications.getExpoPushTokenAsync()).data;
    console.log('Expo Push Token:', token);
    return token;
  } else {
    Alert.alert('Must use physical device for Push Notifications');
  }
};

const sendPushNotification = async (expoPushToken) => {
  const message = {
    to: expoPushToken,
    sound: 'default',
    title: 'Test Notification',
    body: 'This is a test message!',
    data: { extraData: 'Some extra data' },
  };

  await fetch('https://exp.host/--/api/v2/push/send', {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(message),
  });
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  title: {
    fontSize: 20,
    marginBottom: 20,
  },
});

export default App;
```

---

### Key Points to Remember
1. **Physical Device**: Push notifications work only on physical devices.
2. **Expo Push Token**: Required to send notifications via Expo’s Push Service.
3. **Background Notifications**: To handle background notifications, implement logic on the server side.
4. **Custom Notifications**: Use the Expo Push Notification Tool for debugging and testing.

By following these steps, you can set up and test push notifications in your Expo-based React Native project.