A recursive descent parser is a top-down parser that uses a set of recursive procedures to process the input. Below, I'll provide a C++ implementation of a simple recursive descent parser for arithmetic expressions supporting the basic operators: `+`, `-`, `*`, and `/`.

### Project Structure

You can structure your project as follows:

```
recursive_descent_parser/
├── CMakeLists.txt
└── src/
    ├── main.cpp
    └── parser.cpp
```

### Step 1: CMakeLists.txt

Here’s a simple `CMakeLists.txt` file to set up the project:

```cmake
cmake_minimum_required(VERSION 3.10)
project(RecursiveDescentParser)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(parser src/main.cpp src/parser.cpp)
```

### Step 2: Parser Implementation (parser.cpp)

Create the `parser.cpp` file with the implementation of the parser:

```cpp
#include <iostream>
#include <sstream>
#include <cctype>
#include <stdexcept>

class Parser {
public:
    explicit Parser(const std::string &input) : input(input), pos(0) {
        currentChar = input[pos];
    }

    double parse() {
        return expression();
    }

private:
    std::string input;
    size_t pos;
    char currentChar;

    void advance() {
        pos++;
        if (pos < input.length()) {
            currentChar = input[pos];
        } else {
            currentChar = '\0'; // End of input
        }
    }

    void skipWhitespace() {
        while (currentChar != '\0' && std::isspace(currentChar)) {
            advance();
        }
    }

    double expression() {
        double result = term();
        while (currentChar == '+' || currentChar == '-') {
            char op = currentChar;
            advance();
            if (op == '+') {
                result += term();
            } else if (op == '-') {
                result -= term();
            }
        }
        return result;
    }

    double term() {
        double result = factor();
        while (currentChar == '*' || currentChar == '/') {
            char op = currentChar;
            advance();
            if (op == '*') {
                result *= factor();
            } else if (op == '/') {
                result /= factor();
            }
        }
        return result;
    }

    double factor() {
        skipWhitespace();
        double result;
        if (std::isdigit(currentChar)) {
            result = number();
        } else if (currentChar == '(') {
            advance();
            result = expression();
            if (currentChar != ')') {
                throw std::runtime_error("Expected ')'");
            }
            advance();
        } else {
            throw std::runtime_error("Unexpected character: " + std::string(1, currentChar));
        }
        skipWhitespace();
        return result;
    }

    double number() {
        std::string numberStr;
        while (std::isdigit(currentChar) || currentChar == '.') {
            numberStr += currentChar;
            advance();
        }
        return std::stod(numberStr);
    }
};
```

### Step 3: Main Program (main.cpp)

Create the `main.cpp` file to test the parser:

```cpp
#include <iostream>
#include "parser.cpp"

int main() {
    std::string input;
    std::cout << "Enter an arithmetic expression: ";
    std::getline(std::cin, input);

    try {
        Parser parser(input);
        double result = parser.parse();
        std::cout << "Result: " << result << std::endl;
    } catch (const std::runtime_error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
```

### Step 4: Build and Run the Project

1. Open a terminal and navigate to the project root directory (`recursive_descent_parser`).
2. Create the build directory and navigate into it:
   ```bash
   mkdir build
   cd build
   ```
3. Run CMake to generate the build files:
   ```bash
   cmake ..
   ```
4. Build the project:
   ```bash
   make
   ```
5. Run the parser:
   ```bash
   ./parser
   ```

### Example Input and Output

When you run the program, you can enter arithmetic expressions such as:

```
Enter an arithmetic expression: 3 + 5 * (2 - 8)
Result: -13
```

### Explanation of the Code

1. **Parser Class**: The `Parser` class initializes with the input string and manages the parsing state.
2. **parse() Method**: The entry point that begins the parsing process.
3. **advance() Method**: Moves to the next character in the input.
4. **skipWhitespace() Method**: Skips over any whitespace characters.
5. **expression() Method**: Handles addition and subtraction.
6. **term() Method**: Handles multiplication and division.
7. **factor() Method**: Handles numbers and parentheses.
8. **number() Method**: Parses numeric values (integers and decimals).

### Conclusion

This implementation of a recursive descent parser provides a straightforward way to parse and evaluate arithmetic expressions. The structure is modular, allowing easy extensions to support more operators or features like unary operators, functions, or variable assignments.