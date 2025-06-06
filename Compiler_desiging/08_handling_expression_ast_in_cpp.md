To evaluate arithmetic expressions represented by an Abstract Syntax Tree (AST) in C++, you can implement a tree traversal method. The most common traversal for evaluating an expression tree is a post-order traversal, where you first evaluate the left and right subtrees before applying the operator at the current node. Below, I will demonstrate how to extend the AST implementation from the previous example to include evaluation capabilities.

### Step 1: Extend the AST Classes

You will add a new method to the `Visitor` class that handles the evaluation of the nodes. The updated `ast.h` file will look like this:

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
    virtual int evaluate(ASTNode &node) = 0; // New evaluate method
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

// Visitor to print the AST
class PrintVisitor : public Visitor {
public:
    void visit(ASTNode &node) override {
        if (auto *numberNode = dynamic_cast<NumberNode*>(&node)) {
            printNumber(*numberNode);
        } else if (auto *binaryOpNode = dynamic_cast<BinaryOpNode*>(&node)) {
            printBinaryOp(*binaryOpNode);
        }
    }

    int evaluate(ASTNode &node) override {
        if (auto *numberNode = dynamic_cast<NumberNode*>(&node)) {
            return numberNode->getValue();
        } else if (auto *binaryOpNode = dynamic_cast<BinaryOpNode*>(&node)) {
            return evaluateBinaryOp(*binaryOpNode);
        }
        throw std::runtime_error("Invalid AST node for evaluation");
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

    int evaluateBinaryOp(BinaryOpNode &node) {
        int leftValue = evaluate(*node.getLeft());
        int rightValue = evaluate(*node.getRight());

        if (node.getOperator() == "+") {
            return leftValue + rightValue;
        } else if (node.getOperator() == "-") {
            return leftValue - rightValue;
        } else if (node.getOperator() == "*") {
            return leftValue * rightValue;
        } else if (node.getOperator() == "/") {
            if (rightValue == 0) {
                throw std::runtime_error("Division by zero");
            }
            return leftValue / rightValue;
        }

        throw std::runtime_error("Invalid binary operator");
    }
};

#endif // AST_H
```

### Step 2: Update the Main Program (main.cpp)

Now, update the `main.cpp` file to use the new evaluation functionality:

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
        
        // Print the AST
        std::cout << "AST: ";
        ast->accept(visitor);
        std::cout << std::endl; // New line after printing the AST
        
        // Evaluate the expression
        int result = visitor.evaluate(*ast);
        std::cout << "Result: " << result << std::endl;
        
    } catch (const std::runtime_error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
```

### Step 3: Build and Run the Project

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

When you run the program, you can input a simple arithmetic expression like `3 + 5 * (2 - 8)`:

```
Enter an expression: 3 + 5 * (2 - 8)
AST: (3 + (5 * (2 - 8)))
Result: -22
```

### How Tree Traversal Can Be Used for Code Interpretation

Tree traversal is a fundamental technique in interpreting and compiling code. Here are some key points about how it works:

1. **Evaluation**: As demonstrated, the AST can be traversed to evaluate expressions by recursively calculating the values of sub-expressions.

2. **Interpretation**: In a language interpreter, traversing an AST allows you to execute code directly. Each node corresponds to a computation or operation that the interpreter performs.

3. **Compilation**: In compilers, traversal can generate intermediate code, machine code, or bytecode from the AST. For each node, the traversal can produce corresponding instructions for the target architecture.

4. **Code Analysis**: Tree traversal can also be used for semantic analysis, type checking, and optimization. Analyzers can traverse the tree to ensure that the code adheres to language rules and best practices.

5. **Code Generation**: For languages that compile to other forms (like C to machine code), traversing the AST can facilitate the generation of output in a structured and logical way.

By utilizing tree traversal, the interpretation and compilation processes become systematic and organized, leading to clearer and more maintainable code.