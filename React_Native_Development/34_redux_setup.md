### Notes on Configuring Redux in a React Native App for Efficient Global State Management

**Objective:**  
Set up Redux in a React Native app to manage global state efficiently.

---

### 1. **What is Redux?**  
Redux is a state management library that helps manage global state in applications. It is particularly useful for managing shared states such as authentication, user preferences, and API data.

---

### 2. **Core Redux Concepts**  
- **Store:** The central repository of the application state.
- **Actions:** Plain JavaScript objects that describe what happened.
- **Reducers:** Pure functions that specify how the application's state changes in response to actions.

---

### 3. **Setting Up Redux in a React Native App**

#### Step 1: Install Redux and Related Libraries

Run the following command to install `redux`, `react-redux`, and `@reduxjs/toolkit`:

```bash
npm install redux react-redux @reduxjs/toolkit
```

---

#### Step 2: Create the Redux Store

Using Redux Toolkit’s `configureStore`, create a centralized store for the application:

**File:** `src/store/index.js`

```javascript
import { configureStore } from '@reduxjs/toolkit';
import counterReducer from '../features/counterSlice';

const store = configureStore({
  reducer: {
    counter: counterReducer,
  },
});

export default store;
```

---

#### Step 3: Define a Slice (Reducer + Actions)

Use Redux Toolkit's `createSlice` to define a slice of the state.

**File:** `src/features/counterSlice.js`

```javascript
import { createSlice } from '@reduxjs/toolkit';

const counterSlice = createSlice({
  name: 'counter',
  initialState: { value: 0 },
  reducers: {
    increment: (state) => {
      state.value += 1;
    },
    decrement: (state) => {
      state.value -= 1;
    },
    incrementByAmount: (state, action) => {
      state.value += action.payload;
    },
  },
});

export const { increment, decrement, incrementByAmount } = counterSlice.actions;

export default counterSlice.reducer;
```

---

#### Step 4: Provide the Store to the Application

Wrap your application with the `Provider` component from `react-redux` to make the store available to all components.

**File:** `App.js`

```javascript
import React from 'react';
import { Provider } from 'react-redux';
import store from './src/store';
import Counter from './src/components/Counter';

const App = () => {
  return (
    <Provider store={store}>
      <Counter />
    </Provider>
  );
};

export default App;
```

---

#### Step 5: Connect Components to the Store

Use the `useSelector` and `useDispatch` hooks from `react-redux` to interact with the store.

**File:** `src/components/Counter.js`

```javascript
import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { increment, decrement, incrementByAmount } from '../features/counterSlice';
import { View, Text, Button, StyleSheet } from 'react-native';

const Counter = () => {
  const count = useSelector((state) => state.counter.value);
  const dispatch = useDispatch();

  return (
    <View style={styles.container}>
      <Text style={styles.counterText}>Count: {count}</Text>
      <Button title="Increment" onPress={() => dispatch(increment())} />
      <Button title="Decrement" onPress={() => dispatch(decrement())} />
      <Button title="Add 5" onPress={() => dispatch(incrementByAmount(5))} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  counterText: {
    fontSize: 24,
    marginBottom: 20,
  },
});

export default Counter;
```

---

### 4. **Efficient State Management with Redux Toolkit**

- **Immer Integration:** Redux Toolkit uses `Immer.js` to allow state mutation in reducers safely.
- **DevTools Integration:** Redux DevTools works out of the box for debugging.
- **Thunk Middleware:** Built-in support for handling async logic using `redux-thunk`.

---

### 5. **Best Practices**

1. **Modular Slices:** Organize slices into feature-based folders for better maintainability.
2. **Avoid Overfetching:** Use selectors to derive state efficiently and avoid redundant API calls.
3. **Middleware:** Leverage middleware like `redux-thunk` or `redux-saga` for side effects (e.g., API calls).
4. **Normalize Data:** Use libraries like `normalizr` to manage complex nested states.
5. **Memoization:** Use `useSelector` with memoized selectors to prevent unnecessary re-renders.

---

### 6. **Common Use Cases for Redux in React Native**

- **Authentication State:** Manage login/logout and user session state.
- **API Data Management:** Store and share fetched data across the app.
- **Theme Management:** Toggle between light and dark modes globally.
- **Cart State in E-commerce Apps:** Manage items in a shopping cart.

---

### 7. **Complete Folder Structure**

```plaintext
src/
├── components/
│   └── Counter.js
├── features/
│   └── counterSlice.js
├── store/
│   └── index.js
```

---

### 8. **Conclusion**

Redux provides a predictable way to manage state, especially for medium-to-large-scale applications. With Redux Toolkit, setting up and managing global state becomes streamlined and efficient in React Native.