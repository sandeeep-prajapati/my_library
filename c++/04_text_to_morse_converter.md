Here's a **complete C++ implementation** for a text-to-Morse and Morse-to-text converter with CLI interface, supporting both English letters/numbers and basic punctuation:

---

### 📻 **Morse Code Converter**
**Features:**
- Convert text → Morse code (with `/` as word separator)
- Convert Morse → text (handles spaces and slashes)
- Validate Morse code input
- Case-insensitive text input

---

## 📁 **Step 1: Project Structure**
```bash
MorseConverter/
├── CMakeLists.txt
├── include/
│   └── Morse.hpp
├── src/
│   ├── main.cpp
│   └── Morse.cpp
└── tests/          # Unit tests (optional)
```

---

## 📜 **Step 2: Morse Code Mapping**
### `include/Morse.hpp`
```cpp
#pragma once
#include <string>
#include <unordered_map>

class MorseConverter {
public:
    static std::string textToMorse(const std::string& text);
    static std::string morseToText(const std::string& morse);

private:
    static const std::unordered_map<char, std::string> _charToMorse;
    static const std::unordered_map<std::string, char> _morseToChar;
    
    static void _initializeMaps();
    static bool _isValidMorse(const std::string& morse);
};
```

### `src/Morse.cpp` (Partial)
```cpp
#include "Morse.hpp"
#include <algorithm>
#include <sstream>

// Character → Morse mapping
const std::unordered_map<char, std::string> MorseConverter::_charToMorse = {
    {'A', ".-"}, {'B', "-..."}, {'C', "-.-."}, {'D', "-.."}, {'E', "."},
    {'F', "..-."}, {'G', "--."}, {'H', "...."}, {'I', ".."}, {'J', ".---"},
    {'K', "-.-"}, {'L', ".-.."}, {'M', "--"}, {'N', "-."}, {'O', "---"},
    {'P', ".--."}, {'Q', "--.-"}, {'R', ".-."}, {'S', "..."}, {'T', "-"},
    {'U', "..-"}, {'V', "...-"}, {'W', ".--"}, {'X', "-..-"}, {'Y', "-.--"},
    {'Z', "--.."},
    {'0', "-----"}, {'1', ".----"}, {'2', "..---"}, {'3', "...--"},
    {'4', "....-"}, {'5', "....."}, {'6', "-...."}, {'7', "--..."},
    {'8', "---.."}, {'9', "----."},
    {'.', ".-.-.-"}, {',', "--..--"}, {'?', "..--.."}, {'\'', ".----."},
    {'!', "-.-.--"}, {'/', "-..-."}, {'(', "-.--."}, {')', "-.--.-"},
    {'&', ".-..."}, {':', "---..."}, {';', "-.-.-."}, {'=', "-...-"},
    {'+', ".-.-."}, {'-', "-....-"}, {'_', "..--.-"}, {'"', ".-..-."},
    {'$', "...-..-"}, {'@', ".--.-."}, {' ', "/"}  // Space between words
};

// Initialize reverse map (Morse → Character)
void MorseConverter::_initializeMaps() {
    static bool initialized = false;
    if (!initialized) {
        for (const auto& pair : _charToMorse) {
            _morseToChar[pair.second] = pair.first;
        }
        initialized = true;
    }
}
```

---

## 🔤 **Step 3: Conversion Logic**
### Text → Morse
```cpp
std::string MorseConverter::textToMorse(const std::string& text) {
    std::stringstream result;
    for (char c : text) {
        char upper = toupper(c);
        if (_charToMorse.find(upper) != _charToMorse.end()) {
            result << _charToMorse.at(upper) << " ";
        } else if (c == ' ') {
            result << "/ ";
        }
    }
    return result.str();
}
```

### Morse → Text
```cpp
std::string MorseConverter::morseToText(const std::string& morse) {
    _initializeMaps();
    std::stringstream result;
    std::stringstream ss(morse);
    std::string token;
    
    while (ss >> token) {
        if (token == "/") {
            result << " ";
        } else if (_morseToChar.find(token) != _morseToChar.end()) {
            result << _morseToChar.at(token);
        } else {
            throw std::invalid_argument("Invalid Morse code: " + token);
        }
    }
    return result.str();
}
```

---

## ⌨️ **Step 4: CLI Interface**
### `src/main.cpp`
```cpp
#include "Morse.hpp"
#include <iostream>

int main() {
    std::cout << "Morse Code Converter\n"
              << "1. Text → Morse\n"
              << "2. Morse → Text\n"
              << "Choice: ";
    
    int choice;
    std::cin >> choice;
    std::cin.ignore(); // Clear newline
    
    std::string input, output;
    std::cout << "Enter input:\n";
    std::getline(std::cin, input);
    
    try {
        if (choice == 1) {
            output = MorseConverter::textToMorse(input);
            std::cout << "Morse code:\n" << output << "\n";
        } else if (choice == 2) {
            output = MorseConverter::morseToText(input);
            std::cout << "Text:\n" << output << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    
    return 0;
}
```

---

## 🔨 **Step 5: Build & Run**
### `CMakeLists.txt`
```cmake
cmake_minimum_required(VERSION 3.10)
project(MorseConverter)

set(CMAKE_CXX_STANDARD 17)

add_executable(morse_converter
    src/main.cpp
    src/Morse.cpp
)
```

### **Commands:**
```bash
mkdir build && cd build
cmake .. && make
./morse_converter
```

---

## 🎯 **Example Usage**
### **Text → Morse:**
```
Input:  SOS! Hello World
Output: ... --- ... -.-.-- / .... . .-.. .-.. --- / .-- --- .-. .-.. -..
```

### **Morse → Text:**
```
Input:  -.-. .- - / .. -. / - .... . / .... .- - 
Output: CAT IN THE HAT
```

---

## 🚀 **Advanced Features**
1. **Sound Generation**  
   Use Beep (Windows) or ASCII Bell (`\a`) for audible Morse:
   ```cpp
   void playMorse(const std::string& morse) {
       for (char c : morse) {
           if (c == '.') Beep(800, 200);  // Short beep
           else if (c == '-') Beep(800, 600); // Long beep
           else Sleep(200); // Pause
       }
   }
   ```

2. **File I/O**  
   Add options to read/write Morse code from files:
   ```cpp
   void convertFile(const std::string& inputFile, bool toMorse);
   ```

3. **WebSocket Server**  
   Create a real-time Morse chat server.

4. **Learning Mode**  
   Quiz users on Morse code translations.

---

### 🔍 **Why This Matters**
- Teaches **map data structures** for bidirectional lookups
- Demonstrates **text processing** and **input validation**
- Foundation for **communication protocols** or **accessibility tools**

Would you like me to add any of these advanced features in detail? Or perhaps a GUI version using Qt?