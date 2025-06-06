### **ESP8266 and WebSockets**

#### 1. **Introduction**
- WebSockets provide a full-duplex communication channel over a single, long-lived connection, enabling real-time communication between clients and servers.
- They are ideal for applications where low latency and high-frequency data updates are necessary, such as chat applications, live notifications, or IoT device monitoring.

#### 2. **Requirements**
- **Hardware**: ESP8266 module (e.g., NodeMCU, Wemos D1 Mini).
- **Software**: Arduino IDE with the ESP8266 board package installed.
- **WebSocket Library**: The `WebSockets` library for Arduino, which can be installed via the Library Manager.

#### 3. **Installing Required Libraries**
- To work with WebSockets, you need to install the `WebSockets` library:
  - **Installation Steps**:
    1. Open the Arduino IDE.
    2. Go to **Sketch** > **Include Library** > **Manage Libraries**.
    3. Search for "WebSockets" and install the library.

#### 4. **Setting Up a WebSocket Server on ESP8266**
- Below is an example of how to set up a WebSocket server on the ESP8266 that communicates with a WebSocket client.

##### 1. **Example Code for WebSocket Server**
```cpp
#include <ESP8266WiFi.h>
#include <WebSocketsServer.h>

// Replace with your network credentials
const char* ssid = "YOUR_SSID";
const char* password = "YOUR_PASSWORD";

WebSocketsServer webSocket = WebSocketsServer(81); // Set WebSocket server on port 81

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  // Wait for the connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected to Wi-Fi!");

  webSocket.begin(); // Start the WebSocket server
  webSocket.onEvent(webSocketEvent); // Register event handler
}

void loop() {
  webSocket.loop(); // Keep the WebSocket server running
}

// WebSocket event handler
void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      Serial.printf("Client %u disconnected\n", num);
      break;
    case WStype_CONNECTED: {
      IPAddress ip = webSocket.remoteIP(num);
      Serial.printf("Client %u connected from %s\n", num, ip.toString().c_str());
      break;
    }
    case WStype_TEXT:
      Serial.printf("Message from client %u: %s\n", num, payload);
      // Echo the received message back to the client
      webSocket.sendTXT(num, "Message received");
      break;
  }
}
```

#### 5. **Explaining the WebSocket Server Code**
- **Include Libraries**: The code includes `ESP8266WiFi.h` for Wi-Fi connectivity and `WebSocketsServer.h` for WebSocket functionality.
- **Define Wi-Fi Credentials**: Replace `YOUR_SSID` and `YOUR_PASSWORD` with your actual Wi-Fi credentials.
- **WebSocket Server Initialization**: An instance of `WebSocketsServer` is created on port 81.
- **Connecting to Wi-Fi**: The ESP8266 connects to the specified Wi-Fi network.
- **WebSocket Setup**:
  - The `webSocket.begin()` function starts the WebSocket server.
  - The `webSocket.onEvent()` function registers the event handler for WebSocket events.
- **Event Handler**:
  - `WStype_DISCONNECTED`: Logs when a client disconnects.
  - `WStype_CONNECTED`: Logs when a client connects, along with the client's IP address.
  - `WStype_TEXT`: Handles incoming text messages and echoes a response back to the client.

#### 6. **Setting Up a WebSocket Client**
- Below is an example of how to set up a WebSocket client using HTML and JavaScript that connects to the ESP8266 WebSocket server.

##### 1. **Example HTML Client Code**
```html
<!DOCTYPE html>
<html>
<head>
  <title>WebSocket Client</title>
</head>
<body>
  <h1>WebSocket Client</h1>
  <button onclick="sendMessage()">Send Message</button>
  <div id="messages"></div>

  <script>
    var socket = new WebSocket('ws://YOUR_ESP8266_IP:81'); // Replace with your ESP8266 IP address

    socket.onopen = function(event) {
      console.log('WebSocket is open now.');
    };

    socket.onmessage = function(event) {
      var messagesDiv = document.getElementById('messages');
      messagesDiv.innerHTML += 'Received: ' + event.data + '<br>';
    };

    socket.onclose = function(event) {
      console.log('WebSocket is closed now.');
    };

    function sendMessage() {
      var message = "Hello from Client!";
      socket.send(message);
      console.log('Sent: ' + message);
    }
  </script>
</body>
</html>
```

#### 7. **Explaining the WebSocket Client Code**
- **HTML Structure**: Basic HTML with a button to send messages and a div to display received messages.
- **WebSocket Connection**: The client connects to the WebSocket server using the ESP8266's IP address.
- **Event Handlers**:
  - `onopen`: Logs when the connection is established.
  - `onmessage`: Displays received messages in the HTML.
  - `onclose`: Logs when the connection is closed.
- **Sending Messages**: The `sendMessage()` function sends a message to the server when the button is clicked.

#### 8. **Conclusion**
- The ESP8266 can implement WebSockets to enable real-time communication with clients, making it suitable for interactive applications.
- This capability enhances the ESP8266's versatility in IoT projects, allowing for instant updates and feedback.
