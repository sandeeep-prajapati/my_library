Here’s a **step-by-step professional guide** to building a **file encryption/decryption tool** in C++ using **AES (via Crypto++ library)** and **XOR** as a simpler alternative. This project will teach you file I/O, cryptography, and command-line argument handling.

---

### **🔒 Project: File Encryptor/Decryptor Tool**  
**Tech Stack:**  
- C++17  
- Crypto++ library (for AES)  
- CMake (for build automation)  
- Git (version control)  

---

### **📂 Step 1: Setup Development Environment**  
#### **Install Dependencies**  
1. **Crypto++ Installation**  
   - **Linux (Debian/Ubuntu):**  
     ```bash
     sudo apt install libcrypto++-dev libcrypto++-utils
     ```
   - **macOS (Homebrew):**  
     ```bash
     brew install cryptopp
     ```
   - **Windows (vcpkg):**  
     ```bash
     vcpkg install cryptopp
     ```

2. **CMake** (Build system)  
   ```bash
   sudo apt install cmake  # Linux/macOS
   ```

---

### **📝 Step 2: Project Structure**  
```plaintext
FileEncryptor/  
├── CMakeLists.txt          # Build configuration  
├── include/  
│   └── crypto_utils.hpp   # Header for crypto functions  
├── src/  
│   ├── main.cpp           # CLI handling  
│   └── crypto_utils.cpp   # AES/XOR implementation  
└── tests/                 # Unit tests (optional)  
```

---

### **🔧 Step 3: Implement AES Encryption (Using Crypto++)**  
#### **`include/crypto_utils.hpp`**  
```cpp
#pragma once
#include <string>

void aes_encrypt_file(const std::string& input_file, 
                      const std::string& output_file, 
                      const std::string& key);

void aes_decrypt_file(const std::string& input_file, 
                      const std::string& output_file, 
                      const std::string& key);
```

#### **`src/crypto_utils.cpp`**  
```cpp
#include "crypto_utils.hpp"
#include <cryptopp/aes.h>
#include <cryptopp/modes.h>
#include <cryptopp/filters.h>
#include <cryptopp/files.h>
#include <cryptopp/hex.h>
#include <stdexcept>

void aes_encrypt_file(const std::string& input_file, 
                      const std::string& output_file, 
                      const std::string& key) {
    using namespace CryptoPP;

    if (key.size() != AES::DEFAULT_KEYLENGTH) {
        throw std::runtime_error("Key must be 16/24/32 bytes long!");
    }

    byte iv[AES::BLOCKSIZE] = {0};  // Initialization Vector (static for simplicity)
    CBC_Mode<AES>::Encryption encryptor(
        reinterpret_cast<const byte*>(key.data()), key.size(), iv);

    FileSource fs(input_file.c_str(), true,
        new StreamTransformationFilter(encryptor,
            new FileSink(output_file.c_str())
        )
    );
}

// Decryption is similar (replace `Encryption` with `Decryption`)
```

---

### **🌀 Step 4: Implement XOR Encryption (Simple Alternative)**  
```cpp
#include <fstream>
#include <vector>

void xor_encrypt_file(const std::string& input_file, 
                      const std::string& output_file, 
                      const std::string& key) {
    std::ifstream in(input_file, std::ios::binary);
    std::ofstream out(output_file, std::ios::binary);
    char c;
    size_t key_index = 0;

    while (in.get(c)) {
        out.put(c ^ key[key_index++ % key.size()]);
    }
}
// XOR decryption is the same function!
```

---

### **🛠 Step 5: CLI Interface (`main.cpp`)**  
```cpp
#include "crypto_utils.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] 
                  << " <encrypt/decrypt> <aes/xor> <input_file> <output_file>\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string method = argv[2];
    std::string input_file = argv[3];
    std::string output_file = argv[4];
    std::string key;

    std::cout << "Enter encryption key: ";
    std::getline(std::cin, key);

    try {
        if (method == "aes") {
            if (mode == "encrypt") aes_encrypt_file(input_file, output_file, key);
            else aes_decrypt_file(input_file, output_file, key);
        } else if (method == "xor") {
            xor_encrypt_file(input_file, output_file, key);
        }
        std::cout << "Operation completed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
```

---

### **🔨 Step 6: Build with CMake**  
#### **`CMakeLists.txt`**  
```cmake
cmake_minimum_required(VERSION 3.10)
project(FileEncryptor)

set(CMAKE_CXX_STANDARD 17)
find_package(cryptopp REQUIRED)

add_executable(file_encryptor
    src/main.cpp
    src/crypto_utils.cpp
)
target_link_libraries(file_encryptor cryptopp)
```

#### **Build & Run**  
```bash
mkdir build && cd build
cmake ..
make
./file_encryptor encrypt aes input.txt output.enc
```

---

### **📌 Key Learnings**  
✅ **File I/O** (Binary file handling)  
✅ **Cryptography** (AES, XOR, Crypto++ usage)  
✅ **Error Handling** (Exceptions for invalid keys)  
✅ **CLI Tools** (Argument parsing)  

---

### **🚀 Next-Level Enhancements**  
1. **Add password-based key derivation (PBKDF2)** for stronger keys.  
2. **Support GCM mode** for authenticated encryption.  
3. **Benchmark performance** (AES vs. XOR).  
4. **Package as a GUI tool** (Qt/ImGui).  

Would you like a **deep dive into AES internals** or **how to sell this tool**? 😊