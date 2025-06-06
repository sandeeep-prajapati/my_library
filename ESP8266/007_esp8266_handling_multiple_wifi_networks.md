### **Handling Multiple Wi-Fi Networks**

#### 1. **Introduction**
- The ESP8266 can connect to multiple Wi-Fi networks, which allows it to switch between available networks based on signal strength or connection quality.
- This capability is useful for IoT applications where maintaining a reliable connection is critical, especially in environments with varying signal strengths.

#### 2. **Storing Multiple Wi-Fi Credentials**
- The ESP8266 allows you to store multiple Wi-Fi credentials (SSIDs and passwords) in the flash memory.
- This enables the device to attempt connections to different networks when needed.

#### 3. **Using Wi-Fi Library for Multiple Networks**
- The Arduino Wi-Fi library can manage connections to multiple networks through the `WiFi.begin()` function and connection checks.

##### 1. **Define an Array for Network Credentials**
```cpp
const char* ssid[] = {"Network1", "Network2", "Network3"};
const char* password[] = {"Password1", "Password2", "Password3"};
```
- Store multiple SSIDs and their corresponding passwords in arrays.

##### 2. **Connecting to the Strongest Network**
- Implement a function to scan for available networks and connect to the strongest one.

**Code Example**:
```cpp
#include <ESP8266WiFi.h>

const char* ssid[] = {"Network1", "Network2", "Network3"};
const char* password[] = {"Password1", "Password2", "Password3"};

void setup() {
  Serial.begin(115200);
  delay(10);
  
  // Start Wi-Fi and scan for networks
  WiFi.mode(WIFI_STA);
  Serial.println("Scanning for available networks...");

  int numNetworks = WiFi.scanNetworks();
  int bestSignal = -100; // Start with a low signal value
  int bestIndex = -1;    // Index of the best network

  // Loop through the found networks
  for (int i = 0; i < numNetworks; i++) {
    Serial.print("Network: ");
    Serial.print(WiFi.SSID(i));
    Serial.print(" Signal strength: ");
    Serial.println(WiFi.RSSI(i));
    
    // Check if this network has a stronger signal
    if (WiFi.RSSI(i) > bestSignal) {
      bestSignal = WiFi.RSSI(i);
      bestIndex = i;
    }
  }

  // If a suitable network is found, attempt to connect
  if (bestIndex != -1) {
    Serial.print("Connecting to the strongest network: ");
    Serial.println(WiFi.SSID(bestIndex));
    WiFi.begin(WiFi.SSID(bestIndex), password[bestIndex]);
  } else {
    Serial.println("No available networks found.");
  }

  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("Connected to Wi-Fi!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Implement other functionalities here
}
```

##### 3. **Explaining the Code**
- **Scanning for Networks**: `WiFi.scanNetworks()` scans for available Wi-Fi networks and returns the number of networks found.
- **Finding the Strongest Signal**:
  - Loop through each network found and print its SSID and signal strength (RSSI).
  - Track the network with the highest RSSI value (strongest signal).
- **Connecting to the Strongest Network**: Use the best network's SSID and password to connect using `WiFi.begin()`.
- **Connection Status**: The code waits until the ESP8266 is connected to the Wi-Fi network, printing the IP address once connected.

#### 4. **Handling Connection Drops and Reconnection**
- Implement a routine in the `loop()` function to periodically check the connection status and reconnect if necessary.
```cpp
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Disconnected! Attempting to reconnect...");
    setup(); // Call setup to reconnect
  }
  // Other functionalities...
  delay(10000); // Delay to prevent rapid reconnect attempts
}
```

#### 5. **Conclusion**
- The ESP8266 can intelligently handle multiple Wi-Fi networks by scanning for available networks and connecting to the strongest one.
- This feature enhances the reliability of IoT devices, allowing them to maintain stable connections in varying environments.
