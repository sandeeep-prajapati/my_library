### **ESP8266 as a Web Server**

#### 1. **Introduction**
- The ESP8266 is capable of acting as a web server, allowing it to host web pages and serve content over a local network or the Internet.
- This capability enables users to control devices, read sensor data, and interact with IoT applications through a web interface.

#### 2. **Requirements**
- **Hardware**: ESP8266 module (e.g., NodeMCU, Wemos D1 Mini).
- **Software**: Arduino IDE with the ESP8266 board package installed.

#### 3. **Setting Up the Web Server**
- To create a web server, you will use the `ESP8266WebServer` library included in the ESP8266 core for Arduino.

##### 1. **Installing the ESP8266 Board in Arduino IDE**
- Make sure the ESP8266 board package is installed in the Arduino IDE. Refer to the previous sections for installation instructions if needed.

##### 2. **Basic Web Server Code Example**
- Here’s a simple example of setting up an ESP8266 as a web server that serves a basic HTML page.

```cpp
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

// Replace with your network credentials
const char* ssid = "YOUR_SSID";
const char* password = "YOUR_PASSWORD";

// Create an instance of the server on port 80
ESP8266WebServer server(80);

// HTML code for the web page
const char* htmlContent = R"rawliteral(
<!DOCTYPE HTML>
<html>
<head>
    <title>ESP8266 Web Server</title>
    <style>
        body { font-family: Arial; }
        h1 { color: #333; }
        p { color: #666; }
    </style>
</head>
<body>
    <h1>Hello from ESP8266</h1>
    <p>This is a simple web server example.</p>
</body>
</html>
)rawliteral";

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  
  // Wait for the connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected to Wi-Fi!");

  // Define the route for the root URL
  server.on("/", []() {
    server.send(200, "text/html", htmlContent);
  });

  // Start the server
  server.begin();
  Serial.println("Server started!");
}

void loop() {
  // Handle client requests
  server.handleClient();
}
```

#### 4. **Explaining the Code**
- **Include Libraries**: The code includes `ESP8266WiFi.h` for Wi-Fi connectivity and `ESP8266WebServer.h` for creating a web server.
- **Define Wi-Fi Credentials**: Replace `YOUR_SSID` and `YOUR_PASSWORD` with your actual Wi-Fi network details.
- **Server Initialization**:
  - An instance of the web server is created on port 80 (HTTP).
  - The server is started with `server.begin()`.
- **Serving HTML Content**:
  - The root URL ("/") is defined with `server.on()`, which sends the HTML content as a response.
- **Loop Function**: The `server.handleClient()` function checks for incoming client requests and processes them.

#### 5. **Adding CSS and JavaScript**
- You can enhance the web page by including CSS for styling and JavaScript for interactivity. Here’s an example of how to add a button that can control an LED connected to the ESP8266.

**Updated HTML Content Example**:
```cpp
const char* htmlContent = R"rawliteral(
<!DOCTYPE HTML>
<html>
<head>
    <title>ESP8266 Web Server</title>
    <style>
        body { font-family: Arial; }
        h1 { color: #333; }
        button { padding: 10px 20px; }
    </style>
</head>
<body>
    <h1>Control LED</h1>
    <button onclick="toggleLED()">Toggle LED</button>
    <script>
        function toggleLED() {
            var xhttp = new XMLHttpRequest();
            xhttp.open("GET", "/toggle", true);
            xhttp.send();
        }
    </script>
</body>
</html>
)rawliteral";
```

##### 1. **Handling Button Click**
- Add a new route to handle the LED toggle request:
```cpp
const int ledPin = 2; // Pin where the LED is connected

void setup() {
  // ... (existing setup code)

  pinMode(ledPin, OUTPUT);
  
  // Define the route for toggling the LED
  server.on("/toggle", []() {
    digitalWrite(ledPin, !digitalRead(ledPin)); // Toggle LED state
    server.send(200, "text/plain", "LED Toggled"); // Send a response
  });
}
```

#### 6. **Conclusion**
- The ESP8266 can effectively function as a web server to host web pages using HTML, CSS, and JavaScript.
- This allows for the creation of interactive web applications that can control hardware components, display sensor data, and provide a user-friendly interface for IoT projects.
