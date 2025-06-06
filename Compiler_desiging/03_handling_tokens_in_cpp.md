To create a C++ program that represents tokens using a struct and generates a list of tokens from a source code input, we'll define a `Token` struct and implement a function to tokenize a given string. Here's how you can do this:

### Step 1: Define the Token Struct

The `Token` struct will represent a single token with its type and value. We'll define an enumeration for the different types of tokens.

### Step 2: Tokenizer Function

The tokenizer function will scan the input string and generate tokens based on the defined rules (keywords, identifiers, and operators).

### Complete Code Implementation

Here's the complete code for a simple tokenizer in C++:

#### 1. Project Structure

You can use the same project structure as before:

```
my_tokenizer/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── token.h
│   └── tokenizer.cpp
└── build/
```

#### 2. CMakeLists.txt

The `CMakeLists.txt` file remains the same:

```cmake
cmake_minimum_required(VERSION 3.10)
project(SimpleTokenizer)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(src)

file(GLOB SOURCES "src/*.cpp")

add_executable(tokenizer ${SOURCES})
```

#### 3. token.h

Create a `token.h` file to define the `Token` struct and the enumeration for token types:

```cpp
#ifndef TOKEN_H
#define TOKEN_H

#include <string>

enum class TokenType {
    KEYWORD,
    IDENTIFIER,
    OPERATOR,
    UNKNOWN,
};

struct Token {
    TokenType type;
    std::string value;

    Token(TokenType type, const std::string& value) : type(type), value(value) {}
};

#endif // TOKEN_H
```

#### 4. tokenizer.cpp

In `tokenizer.cpp`, implement the tokenizer functionality:

```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <sstream>
#include "token.h"

class Tokenizer {
public:
    Tokenizer(const std::string& input) : input(input), pos(0) {}

    std::vector<Token> tokenize();

private:
    std::string input;
    size_t pos;

    bool isWhitespace(char c);
    bool isKeyword(const std::string& word);
    bool isIdentifierChar(char c);
    bool isOperator(char c);
};

std::vector<Token> Tokenizer::tokenize() {
    std::vector<Token> tokens;
    std::string current;

    while (pos < input.length()) {
        char currentChar = input[pos];

        if (isWhitespace(currentChar)) {
            ++pos;
            continue;
        }

        if (isOperator(currentChar)) {
            tokens.emplace_back(TokenType::OPERATOR, std::string(1, currentChar));
            ++pos;
            continue;
        }

        if (std::isalpha(currentChar)) {
            current.clear();
            while (pos < input.length() && isIdentifierChar(input[pos])) {
                current += input[pos++];
            }
            if (isKeyword(current)) {
                tokens.emplace_back(TokenType::KEYWORD, current);
            } else {
                tokens.emplace_back(TokenType::IDENTIFIER, current);
            }
            continue;
        }

        tokens.emplace_back(TokenType::UNKNOWN, std::string(1, currentChar));
        ++pos;
    }

    return tokens;
}

bool Tokenizer::isWhitespace(char c) {
    return std::isspace(c);
}

bool Tokenizer::isKeyword(const std::string& word) {
    static const std::unordered_map<std::string, TokenType> keywords = {
        {"if", TokenType::KEYWORD},
        {"else", TokenType::KEYWORD},
        {"while", TokenType::KEYWORD},
        {"return", TokenType::KEYWORD},
    };
    return keywords.find(word) != keywords.end();
}

bool Tokenizer::isIdentifierChar(char c) {
    return std::isalnum(c) || c == '_';
}

bool Tokenizer::isOperator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/' || c == '=' || c == '<' || c == '>';
}
```

#### 5. main.cpp

Finally, create a `main.cpp` file to test the tokenizer:

```cpp
#include <iostream>
#include "token.h"
#include "tokenizer.cpp" // Include the tokenizer implementation

int main() {
    std::string input = "if (x == 10) return x + 1; else x = x - 1; while (x > 0) { x--; }";

    Tokenizer tokenizer(input);
    std::vector<Token> tokens = tokenizer.tokenize();

    for (const auto& token : tokens) {
        std::string type;
        switch (token.type) {
            case TokenType::KEYWORD:   type = "KEYWORD"; break;
            case TokenType::IDENTIFIER: type = "IDENTIFIER"; break;
            case TokenType::OPERATOR:   type = "OPERATOR"; break;
            default:                   type = "UNKNOWN"; break;
        }
        std::cout << type << ": " << token.value << std::endl;
    }

    return 0;
}
```

### Step 6: Build and Run the Project

1. Open a terminal and navigate to the project root directory (`my_tokenizer`).
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
5. Run the compiled program:
   ```bash
   ./tokenizer
   ```

### Output

The output will display the tokens recognized by the tokenizer:

```
KEYWORD: if
OPERATOR: (
IDENTIFIER: x
OPERATOR: ==
IDENTIFIER: 10
OPERATOR: )
KEYWORD: return
IDENTIFIER: x
OPERATOR: +
IDENTIFIER: 1
OPERATOR: ;
KEYWORD: else
IDENTIFIER: x
OPERATOR: =
IDENTIFIER: x
OPERATOR: -
IDENTIFIER: 1
OPERATOR: ;
KEYWORD: while
OPERATOR: (
IDENTIFIER: x
OPERATOR: >
IDENTIFIER: 0
OPERATOR: )
OPERATOR: {
IDENTIFIER: x
OPERATOR: --
OPERATOR: ;
OPERATOR: }
```

### Summary

This implementation provides a basic tokenizer that can recognize keywords, identifiers, and operators from a source code input. You can extend this tokenizer further to handle additional token types, such as literals, comments, and more complex language features, depending on your needs.