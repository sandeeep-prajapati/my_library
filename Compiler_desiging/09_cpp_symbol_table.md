Implementing a symbol table in C++ is a common task in the creation of compilers and interpreters. A symbol table typically maps variable names to their corresponding types and values. Using `std::unordered_map` is a great choice for this purpose due to its average constant time complexity for insertions and lookups.

Below is a simple implementation of a symbol table that allows you to insert variable names with their types and values, look them up, and remove them if necessary.

### Step 1: Define the Symbol Table Class

Here's how you can define a `SymbolTable` class using `std::unordered_map`:

```cpp
#include <iostream>
#include <unordered_map>
#include <string>
#include <variant>

class SymbolTable {
public:
    using ValueType = std::variant<int, double, std::string>; // Support for different types

    // Insert a variable into the symbol table
    void insert(const std::string &name, const std::string &type, ValueType value) {
        symbols[name] = {type, value};
    }

    // Look up a variable by name
    std::pair<std::string, ValueType> lookup(const std::string &name) {
        if (symbols.find(name) != symbols.end()) {
            return symbols[name];
        }
        throw std::runtime_error("Variable not found: " + name);
    }

    // Remove a variable from the symbol table
    void remove(const std::string &name) {
        symbols.erase(name);
    }

    // Print the contents of the symbol table
    void print() const {
        for (const auto &entry : symbols) {
            std::cout << "Variable: " << entry.first << ", Type: " << entry.second.first << ", Value: ";
            printValue(entry.second.second);
        }
    }

private:
    // Store the variable name, type, and value
    std::unordered_map<std::string, std::pair<std::string, ValueType>> symbols;

    // Helper function to print the value based on its type
    void printValue(const ValueType &value) const {
        std::visit([](const auto &val) {
            std::cout << val << std::endl;
        }, value);
    }
};
```

### Step 2: Example Usage

Now let's create a simple main program to demonstrate how to use the `SymbolTable` class:

```cpp
#include <iostream>
#include "SymbolTable.h" // Include the header file

int main() {
    SymbolTable symbolTable;

    // Inserting variables
    symbolTable.insert("x", "int", 10);
    symbolTable.insert("y", "double", 3.14);
    symbolTable.insert("name", "string", std::string("Alice"));

    // Print the symbol table
    std::cout << "Symbol Table:" << std::endl;
    symbolTable.print();

    // Lookup a variable
    try {
        auto [type, value] = symbolTable.lookup("x");
        std::cout << "Lookup x: Type = " << type << ", Value = ";
        symbolTable.printValue(value); // Print the value of x
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    // Remove a variable
    symbolTable.remove("y");
    std::cout << "After removing y:" << std::endl;
    symbolTable.print();

    return 0;
}
```

### Step 3: Compile and Run

To compile the code, create a file named `SymbolTable.cpp` for the implementation and a file named `main.cpp` for the example usage.

1. Open a terminal and create the files:

```bash
touch SymbolTable.cpp main.cpp
```

2. Add the code provided above into the respective files.

3. Compile the program:

```bash
g++ -std=c++17 -o symbol_table_example SymbolTable.cpp main.cpp
```

4. Run the program:

```bash
./symbol_table_example
```

### Expected Output

The output should look similar to this:

```
Symbol Table:
Variable: x, Type: int, Value: 10
Variable: y, Type: double, Value: 3.14
Variable: name, Type: string, Value: Alice
Lookup x: Type = int, Value = 10
After removing y:
Variable: x, Type: int, Value: 10
Variable: name, Type: string, Value: Alice
```

### Explanation of the Code

1. **Symbol Table Class**: The `SymbolTable` class manages the storage of variables using an `unordered_map`, where each variable name maps to a pair containing its type and value.

2. **Value Type**: The `ValueType` is defined using `std::variant`, which allows you to store different data types (e.g., `int`, `double`, `std::string`).

3. **Insertion and Lookup**: The `insert` method allows you to add a variable to the symbol table, while the `lookup` method retrieves the variable's type and value.

4. **Printing the Table**: The `print` method iterates through the map and prints each variable's name, type, and value.

5. **Exception Handling**: If you try to look up a variable that doesn't exist, a `std::runtime_error` is thrown.

This implementation can be extended further to include features such as type checking, scope management, and more, depending on your needs.