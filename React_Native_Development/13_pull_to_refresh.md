Adding pull-to-refresh functionality to a `FlatList` component in React Native is straightforward using the `refreshing` and `onRefresh` props provided by `FlatList`. These allow you to indicate when a refresh is happening and specify what should occur when the user performs a pull-to-refresh gesture.

### Code Example

```javascript
import React, { useState } from 'react';
import { View, FlatList, Text, StyleSheet, RefreshControl } from 'react-native';

const App = () => {
  const [data, setData] = useState([
    { id: '1', title: 'Item 1' },
    { id: '2', title: 'Item 2' },
    { id: '3', title: 'Item 3' },
    { id: '4', title: 'Item 4' },
  ]);

  const [refreshing, setRefreshing] = useState(false);

  const onRefresh = () => {
    // Simulate a network request
    setRefreshing(true);
    setTimeout(() => {
      // Add new data or refresh existing data
      setData((prevData) => [
        { id: `${prevData.length + 1}`, title: `Item ${prevData.length + 1}` },
        ...prevData,
      ]);
      setRefreshing(false);
    }, 1500);
  };

  const renderItem = ({ item }) => (
    <View style={styles.item}>
      <Text style={styles.text}>{item.title}</Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <FlatList
        data={data}
        keyExtractor={(item) => item.id}
        renderItem={renderItem}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  item: {
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#ccc',
  },
  text: {
    fontSize: 18,
  },
});

export default App;
```

---

### Explanation

1. **State Management:**
   - `refreshing`: A state variable that determines if the `FlatList` is currently refreshing.
   - `data`: The state holding the list of items displayed in the `FlatList`.

2. **`onRefresh`:**
   - This function is triggered when the user performs a pull-to-refresh gesture.
   - Inside `onRefresh`, simulate a network request using `setTimeout` to mimic fetching or updating data.
   - Update the `data` state to add or refresh items in the list.

3. **`refreshControl`:**
   - The `RefreshControl` component is passed to the `refreshControl` prop of `FlatList`.
   - It handles the visual indicator for the pull-to-refresh action and binds the `refreshing` state and `onRefresh` function.

4. **FlatList Properties:**
   - `data`: The array of items to render.
   - `renderItem`: A function that renders each item in the list.
   - `keyExtractor`: A unique key for each item to improve performance.

---

### Output

- The app displays a list of items.
- Pulling down the list triggers a refreshing spinner, adds a new item to the top, and stops refreshing after 1.5 seconds.

---

### Enhancements
1. **Error Handling:**
   Add error handling for network requests in the `onRefresh` function.

2. **Loading Indicator:**
   Display a loader while the list is being updated.

3. **Dynamic Data Fetching:**
   Integrate a real API call using libraries like `axios` or `fetch`.

4. **Customization:**
   Customize the `RefreshControl` appearance by modifying its colors:
   ```javascript
   <RefreshControl
     refreshing={refreshing}
     onRefresh={onRefresh}
     colors={['#ff0000']} // For Android
     tintColor="#ff0000"  // For iOS
   />
   ```