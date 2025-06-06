Designing an Abstract Syntax Tree (AST) involves creating a class structure that can represent various components of a parsed expression, such as literals, binary operations, and parenthetical expressions. Below is a C++ implementation for a simple AST to represent arithmetic expressions, along with a parser that constructs the AST and a function to print it.

### Project Structure

You can structure your project as follows:

```
ast_example/
├── CMakeLists.txt
└── src/
    ├── main.cpp
    ├── ast.h
    ├── parser.cpp
    ├── parser.h
    └── lexer.cpp
    └── lexer.h
```

### Step 1: CMakeLists.txt

Here’s a simple `CMakeLists.txt` file to set up the project:

```cmake
cmake_minimum_required(VERSION 3.10)
project(ASTExample)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(ast_example src/main.cpp src/lexer.cpp src/parser.cpp)
```

### Step 2: AST Class Structure (ast.h)

Create the `ast.h` file that defines the classes for the AST nodes:

```cpp
#ifndef AST_H
#define AST_H

#include <iostream>
#include <memory>
#include <string>

// Forward declaration
class ASTNode;

class Visitor {
public:
    virtual void visit(ASTNode &node) = 0;
};

// Base class for all AST nodes
class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual void accept(Visitor &visitor) = 0;
};

// Class for number literals
class NumberNode : public ASTNode {
public:
    explicit NumberNode(int value) : value(value) {}

    void accept(Visitor &visitor) override {
        visitor.visit(*this);
    }

    int getValue() const { return value; }

private:
    int value;
};

// Class for binary operations
class BinaryOpNode : public ASTNode {
public:
    BinaryOpNode(std::unique_ptr<ASTNode> left, std::string op, std::unique_ptr<ASTNode> right)
        : left(std::move(left)), op(std::move(op)), right(std::move(right)) {}

    void accept(Visitor &visitor) override {
        visitor.visit(*this);
    }

    const ASTNode *getLeft() const { return left.get(); }
    const ASTNode *getRight() const { return right.get(); }
    const std::string &getOperator() const { return op; }

private:
    std::unique_ptr<ASTNode> left;
    std::string op;
    std::unique_ptr<ASTNode> right;
};

// A simple visitor that prints the AST
class PrintVisitor : public Visitor {
public:
    void visit(ASTNode &node) override {
        if (dynamic_cast<NumberNode*>(&node)) {
            printNumber(static_cast<NumberNode&>(node));
        } else if (dynamic_cast<BinaryOpNode*>(&node)) {
            printBinaryOp(static_cast<BinaryOpNode&>(node));
        }
    }

private:
    void printNumber(NumberNode &node) {
        std::cout << node.getValue();
    }

    void printBinaryOp(BinaryOpNode &node) {
        std::cout << "(";
        node.getLeft()->accept(*this);
        std::cout << " " << node.getOperator() << " ";
        node.getRight()->accept(*this);
        std::cout << ")";
    }
};

#endif // AST_H
```

### Step 3: Lexer Implementation (lexer.cpp and lexer.h)

Create the `lexer.h` file that defines the lexer interface:

```cpp
#ifndef LEXER_H
#define LEXER_H

#include <string>

enum class Token {
    NUMBER,
    PLUS,
    MINUS,
    MULTIPLY,
    DIVIDE,
    LPAREN,
    RPAREN,
    END,
    INVALID
};

class Lexer {
public:
    explicit Lexer(const std::string &input);
    Token getNextToken();
    int getCurrentNumber() const;

private:
    std::string input;
    size_t pos;
    int currentNumber;
    char currentChar;

    void advance();
    void skipWhitespace();
};

#endif // LEXER_H
```

And create the `lexer.cpp` file with the lexer implementation:

```cpp
#include "lexer.h"
#include <cctype>
#include <stdexcept>

Lexer::Lexer(const std::string &input) : input(input), pos(0) {
    currentChar = input[pos];
}

void Lexer::advance() {
    pos++;
    if (pos < input.length()) {
        currentChar = input[pos];
    } else {
        currentChar = '\0'; // End of input
    }
}

void Lexer::skipWhitespace() {
    while (currentChar != '\0' && std::isspace(currentChar)) {
        advance();
    }
}

Token Lexer::getNextToken() {
    while (currentChar != '\0') {
        if (std::isspace(currentChar)) {
            skipWhitespace();
            continue;
        }

        if (std::isdigit(currentChar)) {
            currentNumber = 0;
            while (std::isdigit(currentChar)) {
                currentNumber = currentNumber * 10 + (currentChar - '0');
                advance();
            }
            return Token::NUMBER;
        }

        if (currentChar == '+') {
            advance();
            return Token::PLUS;
        }

        if (currentChar == '-') {
            advance();
            return Token::MINUS;
        }

        if (currentChar == '*') {
            advance();
            return Token::MULTIPLY;
        }

        if (currentChar == '/') {
            advance();
            return Token::DIVIDE;
        }

        if (currentChar == '(') {
            advance();
            return Token::LPAREN;
        }

        if (currentChar == ')') {
            advance();
            return Token::RPAREN;
        }

        return Token::INVALID; // Unknown character
    }

    return Token::END; // End of input
}

int Lexer::getCurrentNumber() const {
    return currentNumber;
}
```

### Step 4: Parser Implementation (parser.cpp and parser.h)

Create the `parser.h` file that defines the parser interface:

```cpp
#ifndef PARSER_H
#define PARSER_H

#include "lexer.h"
#include "ast.h"
#include <memory>

class Parser {
public:
    explicit Parser(Lexer &lexer) : lexer(lexer), currentToken(lexer.getNextToken()) {}

    std::unique_ptr<ASTNode> parse();

private:
    Lexer &lexer;
    Token currentToken;

    void eat(Token token);
    std::unique_ptr<ASTNode> factor();
    std::unique_ptr<ASTNode> term();
    std::unique_ptr<ASTNode> expression();
};

#endif // PARSER_H
```

And create the `parser.cpp` file with the parser implementation:

```cpp
#include "parser.h"
#include <stdexcept>

void Parser::eat(Token token) {
    if (currentToken == token) {
        currentToken = lexer.getNextToken();
    } else {
        throw std::runtime_error("Unexpected token");
    }
}

std::unique_ptr<ASTNode> Parser::factor() {
    if (currentToken == Token::NUMBER) {
        int value = lexer.getCurrentNumber();
        eat(Token::NUMBER);
        return std::make_unique<NumberNode>(value);
    } else if (currentToken == Token::LPAREN) {
        eat(Token::LPAREN);
        auto node = expression();
        eat(Token::RPAREN);
        return node;
    }
    throw std::runtime_error("Expected number or '('");
}

std::unique_ptr<ASTNode> Parser::term() {
    auto node = factor();
    while (currentToken == Token::MULTIPLY || currentToken == Token::DIVIDE) {
        std::string op = (currentToken == Token::MULTIPLY) ? "*" : "/";
        eat(currentToken);
        node = std::make_unique<BinaryOpNode>(std::move(node), op, factor());
    }
    return node;
}

std::unique_ptr<ASTNode> Parser::expression() {
    auto node = term();
    while (currentToken == Token::PLUS || currentToken == Token::MINUS) {
        std::string op = (currentToken == Token::PLUS) ? "+" : "-";
        eat(currentToken);
        node = std::make_unique<BinaryOpNode>(std::move(node), op, term());
    }
    return node;
}

std::unique_ptr<ASTNode> Parser::parse() {
    return expression();
}
```

### Step 5: Main Program (main.cpp)

Create the `main.cpp` file to run the parser and print the AST:

```cpp
#include <iostream>
#include "lexer.h"
#include "parser.h"

int main() {
    std::string input;
    std::cout << "Enter an expression: ";
    std::getline(std::cin, input);

    Lexer lexer(input);
    Parser parser(lexer);

    try {
        auto ast = parser.parse();
        PrintVisitor visitor;
        ast->accept(visitor);
        std::cout << std::endl; // New line after printing the AST
    } catch (const std::runtime_error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
```

### Step 6: Build and Run the Project

1. Open a terminal and navigate to the project root directory (`ast_example`).
2. Create the build directory and navigate into it:
   ```bash
   mkdir build
   cd build
   ```
3. Run CMake to generate the build files:
   ```bash
   cmake ..


   ```
4. Compile the project:
   ```bash
   make
   ```
5. Run the program:
   ```bash
   ./ast_example
   ```

### Example Usage

When you run the program, you can input a simple arithmetic expression like `3 + 5 * (2 - 8)` and it will output the resulting AST structure in a readable format, such as:

```
Enter an expression: 3 + 5 * (2 - 8)
(3 + (5 * (2 - 8)))
```

This structure allows you to see how the expression is represented in the AST. The implementation can be extended to include more features, such as error handling for invalid expressions and support for additional operators.