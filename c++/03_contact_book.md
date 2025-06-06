Here's a **step-by-step implementation** of a CLI-based contact manager in C++ that stores data in both CSV and binary formats, with features for adding, searching, and exporting contacts:

---

### 📇 **Project: Contact Manager**
**Features:**
- Add/delete/search contacts
- Save in CSV (human-readable) or binary (compact)
- Load from existing files
- Simple command-line interface

---

## 🛠️ **Step 1: Project Structure**
```bash
ContactManager/
├── CMakeLists.txt
├── include/
│   └── ContactManager.hpp
├── src/
│   ├── main.cpp
│   └── ContactManager.cpp
└── samples/               # Example data files
    ├── contacts.csv
    └── contacts.bin
```

---

## 📝 **Step 2: Core Contact Class**
### `include/ContactManager.hpp`
```cpp
#pragma once
#include <string>
#include <vector>

struct Contact {
    std::string name;
    std::string phone;
    std::string email;
    
    // Serialization methods
    void to_csv(std::ostream& os) const;
    void to_binary(std::ostream& os) const;
    static Contact from_csv(std::istream& is);
    static Contact from_binary(std::istream& is);
};

class ContactManager {
public:
    void add(const Contact& contact);
    bool remove(const std::string& name);
    std::vector<Contact> search(const std::string& query) const;
    
    bool save_csv(const std::string& filename) const;
    bool save_binary(const std::string& filename) const;
    bool load_csv(const std::string& filename);
    bool load_binary(const std::string& filename);

private:
    std::vector<Contact> _contacts;
};
```

---

## 💾 **Step 3: File I/O Implementation**
### `src/ContactManager.cpp` (Key Parts)
```cpp
// CSV Serialization
void Contact::to_csv(std::ostream& os) const {
    os << "\"" << name << "\","
       << "\"" << phone << "\","
       << "\"" << email << "\"\n";
}

Contact Contact::from_csv(std::istream& is) {
    Contact c;
    char delim;
    is >> delim; // Read opening "
    std::getline(is, c.name, '"');
    is >> delim >> delim; // Skip ,"
    std::getline(is, c.phone, '"');
    is >> delim >> delim; // Skip ,"
    std::getline(is, c.email, '"');
    return c;
}

// Binary Serialization
void Contact::to_binary(std::ostream& os) const {
    size_t size = name.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(name.c_str(), size);
    
    size = phone.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(phone.c_str(), size);
    
    size = email.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(email.c_str(), size);
}

Contact Contact::from_binary(std::istream& is) {
    Contact c;
    size_t size;
    
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    c.name.resize(size);
    is.read(&c.name[0], size);
    
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    c.phone.resize(size);
    is.read(&c.phone[0], size);
    
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    c.email.resize(size);
    is.read(&c.email[0], size);
    
    return c;
}
```

---

## ⌨️ **Step 4: CLI Interface**
### `src/main.cpp`
```cpp
#include "ContactManager.hpp"
#include <iostream>
#include <fstream>

void print_menu() {
    std::cout << "\nContact Manager\n"
              << "1. Add Contact\n"
              << "2. Search Contacts\n"
              << "3. Save to CSV\n"
              << "4. Save to Binary\n"
              << "5. Load from CSV\n"
              << "6. Load from Binary\n"
              << "0. Exit\n"
              << "> ";
}

int main() {
    ContactManager manager;
    
    while (true) {
        print_menu();
        int choice;
        std::cin >> choice;
        std::cin.ignore(); // Clear newline
        
        if (choice == 1) {
            Contact c;
            std::cout << "Name: "; std::getline(std::cin, c.name);
            std::cout << "Phone: "; std::getline(std::cin, c.phone);
            std::cout << "Email: "; std::getline(std::cin, c.email);
            manager.add(c);
        }
        else if (choice == 2) {
            std::string query;
            std::cout << "Search: "; std::getline(std::cin, query);
            auto results = manager.search(query);
            for (const auto& c : results) {
                std::cout << c.name << " | " << c.phone << " | " << c.email << "\n";
            }
        }
        else if (choice >= 3 && choice <= 6) {
            std::string filename;
            std::cout << "Filename: "; std::getline(std::cin, filename);
            
            bool success = false;
            if (choice == 3) success = manager.save_csv(filename);
            else if (choice == 4) success = manager.save_binary(filename);
            else if (choice == 5) success = manager.load_csv(filename);
            else if (choice == 6) success = manager.load_binary(filename);
            
            std::cout << (success ? "Success!" : "Failed!") << "\n";
        }
        else if (choice == 0) break;
    }
    
    return 0;
}
```

---

## 🔨 **Step 5: Build System**
### `CMakeLists.txt`
```cmake
cmake_minimum_required(VERSION 3.10)
project(ContactManager)

set(CMAKE_CXX_STANDARD 17)

add_executable(contact_manager
    src/main.cpp
    src/ContactManager.cpp
)
```

---

## 📊 **File Format Comparison**
| Feature        | CSV                          | Binary                     |
|----------------|------------------------------|----------------------------|
| Human-readable | ✅ Yes                       | ❌ No                      |
| File size      | Larger (~2x)                 | Compact                    |
| Parsing speed  | Slower (text processing)     | Faster (direct memory read)|
| Compatibility  | Works in Excel/Text editors  | C++ only                   |

---

## 🚀 **Advanced Enhancements**
1. **Encryption**  
   Add AES encryption to binary files using Crypto++:
   ```cpp
   bool save_encrypted(const std::string& filename, const std::string& key);
   ```

2. **Duplicate Detection**  
   Prevent duplicate entries:
   ```cpp
   bool exists(const Contact& c) const;
   ```

3. **Import/Export**  
   Add vCard (.vcf) support:
   ```cpp
   bool export_vcard(const std::string& filename) const;
   ```

4. **Birthday Field**  
   Extend Contact struct with dates:
   ```cpp
   std::chrono::year_month_day birthday;
   ```

5. **SQLite Backend**  
   Replace file I/O with SQL database:
   ```sql
   CREATE TABLE contacts (name TEXT, phone TEXT, email TEXT);
   ```

---

### 💡 **Usage Example**
```bash
# Build and run
mkdir build && cd build
cmake .. && make
./contact_manager

# Sample CSV output
"John Doe","+123456789","john@example.com"
"Jane Smith","+987654321","jane@example.com"
```

Would you like me to elaborate on any specific part? For example:
1. How to add phone number validation?
2. Implementing a GUI version with Qt?
3. Adding compression to binary files?
4. Creating a Python wrapper for the binary format?