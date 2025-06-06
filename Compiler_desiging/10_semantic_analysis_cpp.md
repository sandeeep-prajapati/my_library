Performing semantic checks on an Abstract Syntax Tree (AST) is an essential part of compiling or interpreting a programming language. The checks typically include ensuring that variables are declared before they are used, verifying type consistency in operations, and other language-specific rules.

Here's a basic implementation of a semantic check function in C++. For the purpose of this example, we'll assume a simple AST structure for arithmetic expressions and a few variable declarations.

### Step 1: Define the AST Structure

We'll first define the AST structure, including nodes for variable declarations, assignments, and arithmetic expressions. 

```cpp
#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <memory>

// Forward declarations
struct VariableDeclaration;
struct Variable;
struct Assignment;
struct BinaryOperation;
struct ASTNode;

// Define types for the variant
using ValueType = std::variant<int, double>;
using ASTNodePtr = std::shared_ptr<ASTNode>;

// Base class for AST nodes
struct ASTNode {
    virtual ~ASTNode() = default;
};

// Node for variable declarations
struct VariableDeclaration : public ASTNode {
    std::string name;
    std::string type; // e.g., "int", "double"
};

// Node for variable usage
struct Variable : public ASTNode {
    std::string name;
};

// Node for assignments
struct Assignment : public ASTNode {
    std::shared_ptr<Variable> variable;
    ASTNodePtr value;
};

// Node for binary operations
struct BinaryOperation : public ASTNode {
    ASTNodePtr left;
    ASTNodePtr right;
    std::string op; // e.g., "+", "-", "*", "/"
};
```

### Step 2: Define the Symbol Table

Next, we’ll define a simple symbol table to keep track of variable declarations.

```cpp
class SymbolTable {
public:
    void declare(const std::string &name, const std::string &type) {
        symbols[name] = type;
    }

    bool isDeclared(const std::string &name) const {
        return symbols.find(name) != symbols.end();
    }

    std::string getType(const std::string &name) const {
        return symbols.at(name);
    }

private:
    std::unordered_map<std::string, std::string> symbols;
};
```

### Step 3: Implement the Semantic Check Function

Now we can implement the semantic check function. This function will traverse the AST and perform checks based on the structure we've defined.

```cpp
class SemanticAnalyzer {
public:
    SemanticAnalyzer() : symbolTable() {}

    void analyze(ASTNodePtr node) {
        if (auto varDecl = std::dynamic_pointer_cast<VariableDeclaration>(node)) {
            // Handle variable declaration
            symbolTable.declare(varDecl->name, varDecl->type);
        } else if (auto var = std::dynamic_pointer_cast<Variable>(node)) {
            // Check if variable is declared
            if (!symbolTable.isDeclared(var->name)) {
                throw std::runtime_error("Variable not declared: " + var->name);
            }
        } else if (auto assignment = std::dynamic_pointer_cast<Assignment>(node)) {
            // Check assignment
            if (!symbolTable.isDeclared(assignment->variable->name)) {
                throw std::runtime_error("Variable not declared: " + assignment->variable->name);
            }
            // Analyze value
            analyze(assignment->value);
        } else if (auto binOp = std::dynamic_pointer_cast<BinaryOperation>(node)) {
            // Analyze left and right operands
            analyze(binOp->left);
            analyze(binOp->right);
        }
    }

private:
    SymbolTable symbolTable;
};
```

### Step 4: Example Usage

Here’s how you can use the `SemanticAnalyzer` with some example AST nodes:

```cpp
int main() {
    try {
        // Create a simple AST for the expression: int x; x = 5 + 3;
        auto varDecl = std::make_shared<VariableDeclaration>();
        varDecl->name = "x";
        varDecl->type = "int";

        auto varX = std::make_shared<Variable>();
        varX->name = "x";

        auto five = std::make_shared<BinaryOperation>();
        five->left = std::make_shared<VariableDeclaration>();
        five->right = std::make_shared<Variable>();
        five->op = "+"; // Simplified representation

        auto assignment = std::make_shared<Assignment>();
        assignment->variable = varX;
        assignment->value = five;

        // Create a semantic analyzer
        SemanticAnalyzer analyzer;

        // Analyze the variable declaration
        analyzer.analyze(varDecl);

        // Analyze the assignment
        analyzer.analyze(assignment);

        std::cout << "Semantic analysis passed." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Semantic error: " << e.what() << std::endl;
    }

    return 0;
}
```

### Compilation and Execution

To compile and run the program, save the code in a single file, say `SemanticAnalyzer.cpp`, and compile it using:

```bash
g++ -std=c++17 -o semantic_analyzer SemanticAnalyzer.cpp
./semantic_analyzer
```

### Explanation

1. **AST Structure**: The `ASTNode` structure is the base class for various types of AST nodes, including variable declarations, variable usages, assignments, and binary operations.

2. **Symbol Table**: The `SymbolTable` class manages the declaration and lookup of variable names and their types.

3. **Semantic Analysis**: The `SemanticAnalyzer` class performs checks during the AST traversal:
   - When it encounters a `VariableDeclaration`, it adds the variable to the symbol table.
   - For `Variable` nodes, it checks if they have been declared.
   - For `Assignment` nodes, it ensures the variable being assigned exists in the symbol table and then recursively checks the value being assigned.
   - For `BinaryOperation`, it recursively checks the left and right operands.

4. **Error Handling**: If a variable is used without being declared, an exception is thrown, providing feedback on the semantic error.

This setup provides a foundation for more complex semantic analysis and can be expanded to handle more complex language features, such as type checking and scope management.