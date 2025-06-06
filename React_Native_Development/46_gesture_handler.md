To implement advanced gestures like swipe, pan, and drag in a React Native app, you can use the `react-native-gesture-handler` library. This library allows you to handle complex gestures efficiently and provides smooth interactions like swipe, pan, drag, and more.

Below is a step-by-step guide on how to implement these advanced gestures using `react-native-gesture-handler`.

---

### **1. Install Dependencies**

First, you need to install the necessary packages.

```bash
npm install react-native-gesture-handler react-native-reanimated
```

Then, for iOS, install the CocoaPods dependencies:

```bash
cd ios && pod install && cd ..
```

### **2. Import the Required Components**

In your component file, you’ll need to import the necessary components from `react-native-gesture-handler` and `react-native-reanimated`:

```javascript
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import {
  GestureHandlerRootView,
  GestureDetector,
  Gesture,
  GestureHandler,
  PanGestureHandler,
  Swipeable,
  TouchableOpacity,
} from 'react-native-gesture-handler';
import Animated, { Easing } from 'react-native-reanimated';
```

### **3. Swipe Gesture**

To implement a swipe gesture, you can use the `Swipeable` component or `Gesture` for more control over the animation.

#### **Swipe Gesture Example:**

```javascript
import { Swipeable } from 'react-native-gesture-handler';

const SwipeExample = () => {
  const renderRightActions = () => {
    return (
      <View style={{ justifyContent: 'center', alignItems: 'center', backgroundColor: 'red', width: 75 }}>
        <Text style={{ color: 'white' }}>Delete</Text>
      </View>
    );
  };

  return (
    <Swipeable renderRightActions={renderRightActions}>
      <View style={styles.card}>
        <Text>Swipe to delete</Text>
      </View>
    </Swipeable>
  );
};

const styles = StyleSheet.create({
  card: {
    padding: 20,
    margin: 10,
    backgroundColor: '#ddd',
    borderRadius: 8,
  },
});
```

In the above example, the `Swipeable` component provides built-in swipe functionality and you can customize the actions when the user swipes (e.g., delete or archive).

---

### **4. Pan Gesture (Drag and Move)**

To implement a pan gesture for dragging items, you can use `PanGestureHandler` and `Animated.View` to animate the drag behavior.

#### **Pan Gesture Example:**

```javascript
import React, { useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { PanGestureHandler } from 'react-native-gesture-handler';
import Animated from 'react-native-reanimated';

const DragExample = () => {
  const [dragX, setDragX] = useState(0);
  const [dragY, setDragY] = useState(0);

  const panGestureHandler = Animated.event(
    [{ nativeEvent: { translationX: dragX, translationY: dragY } }],
    { useNativeDriver: false }
  );

  return (
    <View style={styles.container}>
      <PanGestureHandler onGestureEvent={panGestureHandler}>
        <Animated.View
          style={[
            styles.box,
            { transform: [{ translateX: dragX }, { translateY: dragY }] },
          ]}
        >
          <Text style={styles.boxText}>Drag me</Text>
        </Animated.View>
      </PanGestureHandler>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  box: {
    width: 150,
    height: 150,
    backgroundColor: '#4CAF50',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 10,
  },
  boxText: {
    color: 'white',
    fontSize: 18,
  },
});

export default DragExample;
```

In this example, the `PanGestureHandler` allows you to move an element around the screen by tracking its translation on the X and Y axes.

---

### **5. Swipe + Pan Gesture with Animated Movement**

You can combine both swipe and pan gestures for more complex interactions. For example, you might have a draggable element that can also swipe away with a pan gesture.

#### **Swipe + Pan Example:**

```javascript
import React, { useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { PanGestureHandler } from 'react-native-gesture-handler';
import Animated, { Easing } from 'react-native-reanimated';

const SwipePanExample = () => {
  const [dragX, setDragX] = useState(0);
  const [dragY, setDragY] = useState(0);

  const panGestureHandler = Animated.event(
    [{ nativeEvent: { translationX: dragX, translationY: dragY } }],
    { useNativeDriver: false }
  );

  return (
    <View style={styles.container}>
      <PanGestureHandler onGestureEvent={panGestureHandler}>
        <Animated.View
          style={[
            styles.box,
            { transform: [{ translateX: dragX }, { translateY: dragY }] },
          ]}
        >
          <Text style={styles.boxText}>Swipe and Drag me</Text>
        </Animated.View>
      </PanGestureHandler>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  box: {
    width: 150,
    height: 150,
    backgroundColor: '#FF5722',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 10,
  },
  boxText: {
    color: 'white',
    fontSize: 18,
  },
});

export default SwipePanExample;
```

In this example, you can combine the swipe and pan gestures to create more interactive UI elements. The pan gesture handles the dragging, while the swipe gesture (like the `Swipeable` component) can be applied on top for swipe actions.

---

### **6. Conclusion**

With `react-native-gesture-handler`, you can easily implement advanced gestures such as swipe, pan, and drag to create intuitive, interactive UIs in your React Native app. The library allows you to manage gestures with high performance and smooth animations, and you can use it in combination with `react-native-reanimated` for more complex animations and interactions.

Here are some key steps for different gestures:
- **Swipe**: Use the `Swipeable` component or custom swipe gesture handlers.
- **Pan**: Use the `PanGestureHandler` to track translations for drag-and-drop interactions.
- **Combine Gestures**: Use both `PanGestureHandler` and other gestures like swipe or tap for complex interactions.

You can also animate gestures using `react-native-reanimated` for better performance and smoother transitions.