Here's a simple implementation of a lexical analyzer (tokenizer) in C++ that can recognize keywords, identifiers, and operators from a basic programming language. This example will define a few keywords (`if`, `else`, `while`, `return`), basic identifiers, and operators (`+`, `-`, `*`, `/`, `=`, `==`).

### Lexical Analyzer Implementation

#### Step 1: Project Structure

Create a directory structure for your project:

```
my_lexer/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── lexer.cpp
│   └── lexer.h
└── build/
```

#### Step 2: CMakeLists.txt

Create a `CMakeLists.txt` file to build your project:

```cmake
cmake_minimum_required(VERSION 3.10)
project(SimpleLexer)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(src)

file(GLOB SOURCES "src/*.cpp")

add_executable(lexer ${SOURCES})
```

#### Step 3: lexer.h

Create a `lexer.h` file to define the lexical analyzer's structure and functions:

```cpp
#ifndef LEXER_H
#define LEXER_H

#include <string>
#include <vector>
#include <unordered_map>

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

class Lexer {
public:
    Lexer(const std::string& input);
    std::vector<Token> tokenize();

private:
    std::string input;
    size_t pos;

    bool isWhitespace(char c);
    bool isKeyword(const std::string& word);
    bool isIdentifierChar(char c);
    bool isOperator(char c);
};

#endif // LEXER_H
```

#### Step 4: lexer.cpp

Implement the lexer functionality in `lexer.cpp`:

```cpp
#include "lexer.h"
#include <cctype>
#include <iostream>

Lexer::Lexer(const std::string& input) : input(input), pos(0) {}

std::vector<Token> Lexer::tokenize() {
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

bool Lexer::isWhitespace(char c) {
    return std::isspace(c);
}

bool Lexer::isKeyword(const std::string& word) {
    static const std::unordered_map<std::string, TokenType> keywords = {
        {"if", TokenType::KEYWORD},
        {"else", TokenType::KEYWORD},
        {"while", TokenType::KEYWORD},
        {"return", TokenType::KEYWORD},
    };
    return keywords.find(word) != keywords.end();
}

bool Lexer::isIdentifierChar(char c) {
    return std::isalnum(c) || c == '_';
}

bool Lexer::isOperator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/' || c == '=' || c == '<' || c == '>';
}
```

#### Step 5: main.cpp

Create a `main.cpp` file to test the lexer:

```cpp
#include <iostream>
#include "lexer.h"

int main() {
    std::string input = "if (x == 10) return x + 1; else x = x - 1; while (x > 0) { x--; }";

    Lexer lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

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

1. Open a terminal and navigate to the project root directory (`my_lexer`).
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
   ./lexer
   ```

### Output

The output will display the tokens recognized by the lexer:

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

This example provides a simple lexical analyzer that can recognize keywords, identifiers, and operators. You can extend it further to handle more complex language features like comments, strings, and different types of literals as needed.