### Notes for Creating a Bridge to Use Native Functionality in a React Native App

---

#### Prerequisites

1. **Basic Understanding of React Native**: Familiarity with the core concepts of React Native.
2. **Knowledge of Native Development**: Understanding of native development in Android (Java/Kotlin) and iOS (Swift/Objective-C).
3. **Development Setup**:
   - For Android: Android Studio.
   - For iOS: Xcode (macOS required).
4. **Ejected Expo Project**: Bridging requires a bare React Native project or an ejected Expo project.

---

### Key Concepts

A **bridge** in React Native connects JavaScript code to native platform-specific functionality, allowing custom native modules to be exposed to JavaScript.

---

### Steps to Create a Native Module Bridge

#### 1. **Initialize a React Native Project**
If you don’t already have a project:
```bash
npx react-native init NativeBridgeDemo
```

---

#### 2. **Setup Native Modules**
##### Android
1. **Create a Native Module**:
   - Go to `android/app/src/main/java/com/nativebridgedemo/`.
   - Create a new Java file for your module (e.g., `MyNativeModule.java`).

**Code Example for Android Module:**
```java
package com.nativebridgedemo;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Promise;

public class MyNativeModule extends ReactContextBaseJavaModule {
    public MyNativeModule(ReactApplicationContext reactContext) {
        super(reactContext);
    }

    @Override
    public String getName() {
        return "MyNativeModule";
    }

    @ReactMethod
    public void getNativeMessage(Promise promise) {
        try {
            promise.resolve("Hello from Android Native Code!");
        } catch (Exception e) {
            promise.reject("Error", e);
        }
    }
}
```

2. **Register the Module**:
   - Modify `MainApplication.java` to include your module.
   ```java
   @Override
   protected List<ReactPackage> getPackages() {
       return Arrays.<ReactPackage>asList(
           new MainReactPackage(),
           new CustomPackage() // Add this line
       );
   }
   ```

3. **Create a Package for the Module**:
   - Create `CustomPackage.java` to register the module.
   ```java
   package com.nativebridgedemo;

   import com.facebook.react.ReactPackage;
   import com.facebook.react.bridge.NativeModule;
   import com.facebook.react.uimanager.ViewManager;

   import java.util.ArrayList;
   import java.util.Collections;
   import java.util.List;

   public class CustomPackage implements ReactPackage {
       @Override
       public List<NativeModule> createNativeModules(ReactApplicationContext reactContext) {
           List<NativeModule> modules = new ArrayList<>();
           modules.add(new MyNativeModule(reactContext));
           return modules;
       }

       @Override
       public List<ViewManager> createViewManagers(ReactApplicationContext reactContext) {
           return Collections.emptyList();
       }
   }
   ```

---

##### iOS
1. **Create a Native Module**:
   - Go to `ios/NativeBridgeDemo/`.
   - Create a new Swift file (e.g., `MyNativeModule.swift`).

**Code Example for iOS Module:**
```swift
import Foundation

@objc(MyNativeModule)
class MyNativeModule: NSObject {
  @objc
  func getNativeMessage(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    resolve("Hello from iOS Native Code!")
  }

  @objc
  static func requiresMainQueueSetup() -> Bool {
    return false
  }
}
```

2. **Bridge Header Setup**:
   - Create a bridging header (`NativeBridgeDemo-Bridging-Header.h`) to expose Swift to Objective-C:
     ```objc
     #import <React/RCTBridgeModule.h>
     ```

3. **Register the Module**:
   - Modify `AppDelegate.m` to register the new module.

---

#### 3. **Access Native Module in JavaScript**
Create a JavaScript wrapper to expose the native functionality.

**Code Example:**
```javascript
import { NativeModules } from 'react-native';

const { MyNativeModule } = NativeModules;

const getNativeMessage = async () => {
  try {
    const message = await MyNativeModule.getNativeMessage();
    console.log(message);
  } catch (error) {
    console.error(error);
  }
};

export default getNativeMessage;
```

---

#### 4. **Use the Module in Your React Native App**
**Code Example for Integration:**
```javascript
import React, { useEffect } from 'react';
import { View, Text, Button } from 'react-native';
import getNativeMessage from './NativeBridge';

const App = () => {
  const handlePress = async () => {
    const message = await getNativeMessage();
    alert(message);
  };

  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Native Bridge Demo</Text>
      <Button title="Get Native Message" onPress={handlePress} />
    </View>
  );
};

export default App;
```

---

### Notes
1. **Testing**:
   - Test on both Android and iOS devices/emulators to ensure cross-platform compatibility.
2. **Expo Projects**:
   - Bridging requires ejecting the project to bare workflow in Expo.
3. **Async Functions**:
   - Use Promises for handling asynchronous tasks in native modules.
4. **Error Handling**:
   - Always handle potential errors while interacting with native modules.

By following these steps, you can create a bridge to integrate native functionality into your React Native app!