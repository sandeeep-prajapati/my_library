Using environment variables to manage sensitive keys and configurations securely is crucial in protecting your app's secrets and configurations (such as API keys, database credentials, etc.). React Native allows you to manage environment variables securely by using a package like **react-native-config**.

Here are the steps to securely manage sensitive keys and configurations using environment variables in React Native:

---

### **1. Install react-native-config**

`react-native-config` is a library that helps you manage environment variables for React Native projects.

To install it, run the following commands:

```bash
npm install react-native-config
```

For iOS, make sure to install the CocoaPods:

```bash
cd ios && pod install && cd ..
```

---

### **2. Create an `.env` File**

Create an `.env` file in the root of your project. This file will contain all your environment variables.

```bash
// .env
API_KEY=your_api_key_here
FIREBASE_API_KEY=your_firebase_api_key_here
STRIPE_SECRET_KEY=your_stripe_secret_key_here
```

Ensure that the `.env` file is added to your `.gitignore` to prevent it from being tracked by version control:

```bash
// .gitignore
.env
```

---

### **3. Access Environment Variables in Your Code**

Now you can access the environment variables in your React Native code using `Config` from `react-native-config`.

#### Example:

```javascript
// App.js
import React from 'react';
import { Text, View } from 'react-native';
import Config from 'react-native-config';

const App = () => {
  return (
    <View>
      <Text>API Key: {Config.API_KEY}</Text>
      <Text>Firebase API Key: {Config.FIREBASE_API_KEY}</Text>
    </View>
  );
};

export default App;
```

#### **Accessing Specific Environment Variables:**

- `Config.API_KEY` will fetch the value of `API_KEY` from the `.env` file.
- Similarly, you can access other variables like `Config.FIREBASE_API_KEY` and `Config.STRIPE_SECRET_KEY`.

---

### **4. Multiple Environment Configurations**

You might want to have different configurations for different environments (e.g., development, staging, and production). You can achieve this by creating different `.env` files for each environment:

1. `.env` - default settings.
2. `.env.production` - for production-specific settings.
3. `.env.staging` - for staging-specific settings.

Then, configure your app to load the appropriate environment file based on the build variant.

For example:

- `.env.development` for development keys
- `.env.production` for production keys

---

### **5. Use Environment Variables with React Native Build Tools**

React Native supports different build configurations for Android and iOS. You can specify the environment file to be used for each platform in your build process.

#### **Android**

You need to modify your `android/app/build.gradle` file to load the `.env` file when building your app. Add the following code inside the `android` block:

```gradle
// android/app/build.gradle
apply from: project(':react-native-config').projectDir.getPath() + "/dotenv.gradle"
```

Ensure that your `android/build.gradle` has the following configuration to add environment-specific variables:

```gradle
// android/build.gradle
buildTypes {
    release {
        buildConfigField "String", "API_KEY", project.env.API_KEY
    }
}
```

This way, when you build your app, it will load the right environment file based on the environment.

#### **iOS**

For iOS, the process is slightly different. Ensure that the `react-native-config` pod is properly installed by adding it to the `ios/Podfile`:

```ruby
# ios/Podfile
pod 'react-native-config', :path => '../node_modules/react-native-config'
```

Then, run the following to install the CocoaPods dependencies:

```bash
cd ios && pod install && cd ..
```

---

### **6. Access Environment Variables in Native Code**

You can also access the environment variables in native code (Android/iOS) if needed.

#### **Android:**
In Android, the environment variables will be available in the `BuildConfig` class:

```java
String apiKey = BuildConfig.API_KEY;
```

#### **iOS:**
For iOS, the environment variables can be accessed via `Config`:

```objc
NSString *apiKey = [Config API_KEY];
```

---

### **7. Best Practices**

1. **Do Not Hardcode Sensitive Keys**: Always store API keys, tokens, and other sensitive data in `.env` files, not directly in your source code.
2. **Use `.env` Files for Different Environments**: For example, use `.env.production` for production keys and `.env.development` for development keys.
3. **Keep `.env` Files Secure**: Never commit `.env` files to version control. Add them to `.gitignore`.
4. **Validate Environment Variables**: Before using environment variables in your app, ensure that they are defined and valid to avoid runtime errors.

---

### **8. Example Use Case**

Let’s say you're integrating Firebase into your React Native app. You can store the Firebase configuration in environment variables and securely access them in your code.

```bash
// .env
FIREBASE_API_KEY=your_firebase_api_key
FIREBASE_AUTH_DOMAIN=your_firebase_auth_domain
```

Then, in your app:

```javascript
import { initializeApp } from 'firebase/app';
import Config from 'react-native-config';

const firebaseConfig = {
  apiKey: Config.FIREBASE_API_KEY,
  authDomain: Config.FIREBASE_AUTH_DOMAIN,
  // ...other config properties
};

const app = initializeApp(firebaseConfig);
```

---

### **Conclusion**

By using **environment variables** with `react-native-config`, you can securely manage sensitive information in your React Native app without exposing secrets in the codebase. This approach is highly scalable and essential for maintaining security in production apps.