Here’s how to fetch weather data from an API and display it in a React Native app.

---

### Step-by-Step Implementation

#### 1. Choose a Weather API
Use a free weather API like [OpenWeatherMap](https://openweathermap.org/api). Sign up to get an API key.

#### 2. Create a React Native App
Set up a React Native app and install dependencies like `axios` for making API calls.

---

### Full Code Example

```javascript
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  Button,
  StyleSheet,
  ActivityIndicator,
  Keyboard,
} from 'react-native';
import axios from 'axios';

const App = () => {
  const [city, setCity] = useState('');
  const [weatherData, setWeatherData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const API_KEY = 'your_openweathermap_api_key'; // Replace with your API key

  const fetchWeather = async () => {
    if (!city.trim()) {
      setError('Please enter a city name.');
      return;
    }

    setLoading(true);
    setError('');
    Keyboard.dismiss(); // Dismiss the keyboard after search

    try {
      const response = await axios.get(
        `https://api.openweathermap.org/data/2.5/weather`,
        {
          params: {
            q: city,
            appid: API_KEY,
            units: 'metric', // For temperature in Celsius
          },
        }
      );
      setWeatherData(response.data);
    } catch (err) {
      setError('City not found or an error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Weather App</Text>
      <TextInput
        style={styles.input}
        placeholder="Enter city name"
        value={city}
        onChangeText={(text) => setCity(text)}
      />
      <Button title="Get Weather" onPress={fetchWeather} />
      {loading && <ActivityIndicator size="large" color="#0000ff" />}
      {error ? <Text style={styles.errorText}>{error}</Text> : null}
      {weatherData && (
        <View style={styles.weatherContainer}>
          <Text style={styles.weatherText}>
            {weatherData.name}, {weatherData.sys.country}
          </Text>
          <Text style={styles.weatherText}>
            Temperature: {weatherData.main.temp}°C
          </Text>
          <Text style={styles.weatherText}>
            Weather: {weatherData.weather[0].description}
          </Text>
          <Text style={styles.weatherText}>
            Humidity: {weatherData.main.humidity}%
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  input: {
    width: '100%',
    padding: 10,
    borderColor: '#ccc',
    borderWidth: 1,
    borderRadius: 5,
    marginBottom: 10,
    backgroundColor: '#fff',
  },
  weatherContainer: {
    marginTop: 20,
    padding: 15,
    borderRadius: 10,
    backgroundColor: '#e0f7fa',
    alignItems: 'center',
    width: '100%',
  },
  weatherText: {
    fontSize: 18,
    marginVertical: 5,
  },
  errorText: {
    marginTop: 10,
    color: 'red',
    fontSize: 16,
  },
});

export default App;
```

---

### Explanation

1. **API Call with Axios:**
   - The `fetchWeather` function uses `axios` to fetch weather data from OpenWeatherMap based on the city name.

2. **State Management:**
   - `city`: Holds the user-inputted city name.
   - `weatherData`: Stores the fetched weather data.
   - `loading`: Indicates whether the data is being fetched.
   - `error`: Stores error messages if any occur.

3. **Error Handling:**
   - Displays an error message if the city is not found or there’s an issue with the API.

4. **Activity Indicator:**
   - Displays a loading spinner while fetching data.

5. **Displaying Weather Data:**
   - The app shows temperature, weather condition, humidity, and the city name if the data is successfully fetched.

6. **Styling:**
   - Added basic styles for input, weather details, and error messages.

---

### How It Works

1. Enter a city name in the text input field.
2. Tap the "Get Weather" button.
3. The app fetches data from OpenWeatherMap and displays the weather details.

---

### Enhancements

1. **Location-Based Weather:**
   Use the device's GPS to fetch weather for the current location with `expo-location` or similar libraries.

2. **Improved UI:**
   Add icons for weather conditions using libraries like `react-native-vector-icons` or `react-native-weather-icons`.

3. **Cache Data:**
   Store the last fetched weather data using `AsyncStorage`.

4. **Unit Conversion:**
   Add a toggle between Celsius and Fahrenheit.

5. **Daily Forecast:**
   Fetch and display a 7-day weather forecast using the OpenWeatherMap One Call API.

---

This example introduces essential API integration skills and lays the groundwork for more complex React Native projects.