### **ESP8266 and REST APIs**

#### 1. **Introduction**
- REST (Representational State Transfer) APIs allow communication between a client (like the ESP8266) and a server over HTTP. 
- This enables the ESP8266 to send and receive data, making it suitable for various applications, such as fetching weather information, controlling devices, or interacting with cloud services.

#### 2. **Requirements**
- **Hardware**: ESP8266 module (e.g., NodeMCU, Wemos D1 Mini).
- **Software**: Arduino IDE with the ESP8266 board package installed.
- **API Access**: Obtain access to a REST API, such as a weather API (e.g., OpenWeatherMap, WeatherAPI).

#### 3. **Installing Required Libraries**
- For making HTTP requests, you can use the `ESP8266HTTPClient` library, which comes with the ESP8266 core.
- Ensure you have the `ESP8266WiFi` library, which is also included by default.

#### 4. **Consuming a REST API**
- Below is an example of how to send a GET request to a weather API to fetch weather data.

##### 1. **Example Code for Fetching Weather Data**
```cpp
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

// Replace with your network credentials
const char* ssid = "YOUR_SSID";
const char* password = "YOUR_PASSWORD";

// Replace with your weather API endpoint
const char* weatherApi = "http://api.openweathermap.org/data/2.5/weather?q=London&appid=YOUR_API_KEY"; // Replace YOUR_API_KEY

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  // Wait for the connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected to Wi-Fi!");

  // Make HTTP GET request
  HTTPClient http;
  http.begin(weatherApi); // Specify the URL
  int httpResponseCode = http.GET(); // Send the request

  if (httpResponseCode > 0) {
    String payload = http.getString(); // Get the response payload
    Serial.println("Response Code: " + String(httpResponseCode));
    Serial.println("Payload: " + payload);
  } else {
    Serial.println("Error on HTTP request: " + String(httpResponseCode));
  }

  http.end(); // Free resources
}

void loop() {
  // Nothing to do here
}
```

#### 5. **Explaining the Code**
- **Include Libraries**: The code includes `ESP8266WiFi.h` for Wi-Fi connectivity and `ESP8266HTTPClient.h` for making HTTP requests.
- **Define Wi-Fi Credentials and API Endpoint**: Replace `YOUR_SSID`, `YOUR_PASSWORD`, and `YOUR_API_KEY` with your actual Wi-Fi credentials and API key for the weather service.
- **Connect to Wi-Fi**: The ESP8266 connects to the specified Wi-Fi network.
- **HTTP GET Request**:
  - An instance of `HTTPClient` is created to manage the HTTP request.
  - The `http.begin()` function specifies the API endpoint.
  - The `http.GET()` function sends the GET request, and the response code is checked.
- **Handling the Response**:
  - If the request is successful (response code > 0), the response payload is retrieved and printed to the Serial Monitor.
  - The connection is closed with `http.end()`.

#### 6. **Sending Data to a REST API**
- Below is an example of how to send data to a REST API using a POST request.

##### 1. **Example Code for Sending Data**
```cpp
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

// Replace with your network credentials
const char* ssid = "YOUR_SSID";
const char* password = "YOUR_PASSWORD";

// Replace with your API endpoint
const char* apiEndpoint = "http://example.com/api/data"; // Replace with your endpoint

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  // Wait for the connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected to Wi-Fi!");

  // Make HTTP POST request
  HTTPClient http;
  http.begin(apiEndpoint); // Specify the URL
  http.addHeader("Content-Type", "application/json"); // Specify content-type

  // Prepare JSON payload
  String jsonData = "{\"sensor\":\"temperature\", \"value\":25}"; // Example data

  int httpResponseCode = http.POST(jsonData); // Send the request

  if (httpResponseCode > 0) {
    String response = http.getString(); // Get the response payload
    Serial.println("Response Code: " + String(httpResponseCode));
    Serial.println("Response: " + response);
  } else {
    Serial.println("Error on HTTP request: " + String(httpResponseCode));
  }

  http.end(); // Free resources
}

void loop() {
  // Nothing to do here
}
```

#### 7. **Explaining the POST Request Code**
- **Define API Endpoint**: Replace `apiEndpoint` with your actual API endpoint.
- **Prepare JSON Payload**: In this example, a simple JSON object is created to send data.
- **HTTP POST Request**:
  - The `http.addHeader()` function sets the content type to JSON.
  - The `http.POST()` function sends the JSON data to the server.
- **Handling the Response**: Similar to the GET request, the response code and payload are printed to the Serial Monitor.

#### 8. **Conclusion**
- The ESP8266 can effectively consume and send data to REST APIs, making it versatile for IoT applications.
- This functionality allows for integration with various online services, enabling features like data logging, remote control, and real-time monitoring.
