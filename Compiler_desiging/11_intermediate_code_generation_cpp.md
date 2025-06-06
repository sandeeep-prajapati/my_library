Generating intermediate code, such as three-address code (TAC), from an Abstract Syntax Tree (AST) is a key step in the compilation process. Three-address code typically consists of a sequence of instructions that have at most three operands, allowing for simpler translation to machine code.

Here's a simple example of how to generate three-address code from an AST in C++. This example will include structs for AST nodes, TAC instructions, and a code generator class to perform the translation.

### Step 1: Define the AST Structure

We'll start with the same AST structure as in the previous example, which includes nodes for variable declarations, assignments, and binary operations.

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

// Forward declarations
struct VariableDeclaration;
struct Variable;
struct Assignment;
struct BinaryOperation;
struct ASTNode;

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
    std::shared_ptr<ASTNode> value;
};

// Node for binary operations
struct BinaryOperation : public ASTNode {
    std::shared_ptr<ASTNode> left;
    std::shared_ptr<ASTNode> right;
    std::string op; // e.g., "+", "-", "*", "/"
};
```

### Step 2: Define the Three-Address Code (TAC) Structure

Next, we define the structure for intermediate code instructions. Each instruction will have an operator and up to three operands.

```cpp
struct TACInstruction {
    std::string op;      // Operator (e.g., "+", "-", "=", etc.)
    std::string arg1;    // First operand
    std::string arg2;    // Second operand (optional)
    std::string result;  // Result of the operation

    // Print the TAC instruction
    void print() const {
        if (arg2.empty()) {
            std::cout << result << " = " << op << " " << arg1 << std::endl;
        } else {
            std::cout << result << " = " << arg1 << " " << op << " " << arg2 << std::endl;
        }
    }
};
```

### Step 3: Create the Code Generator Class

We will create a code generator that traverses the AST and produces TAC instructions.

```cpp
class CodeGenerator {
public:
    void generate(ASTNode* node) {
        if (auto varDecl = dynamic_cast<VariableDeclaration*>(node)) {
            // Variable declaration: do nothing for TAC
        } else if (auto var = dynamic_cast<Variable*>(node)) {
            // Variable usage: do nothing for TAC
        } else if (auto assignment = dynamic_cast<Assignment*>(node)) {
            // Generate TAC for the assignment
            std::string result = newTemp();
            generate(assignment->value.get());
            tacInstructions.emplace_back(TACInstruction{"=", result, "", assignment->variable->name});
        } else if (auto binOp = dynamic_cast<BinaryOperation*>(node)) {
            // Generate TAC for binary operations
            std::string temp1 = newTemp();
            std::string temp2 = newTemp();
            generate(binOp->left.get());
            generate(binOp->right.get());
            tacInstructions.emplace_back(TACInstruction{binOp->op, temp1, temp2, newTemp()});
        }
    }

    void printTAC() const {
        for (const auto& instruction : tacInstructions) {
            instruction.print();
        }
    }

private:
    std::vector<TACInstruction> tacInstructions;
    int tempCounter = 0;

    // Generate a new temporary variable name
    std::string newTemp() {
        return "t" + std::to_string(tempCounter++);
    }
};
```

### Step 4: Example Usage

Now we can put everything together and demonstrate how to generate three-address code from a simple AST.

```cpp
int main() {
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

    // Create a code generator
    CodeGenerator codeGen;

    // Generate TAC for variable declaration (no TAC generated)
    codeGen.generate(varDecl.get());

    // Generate TAC for assignment
    codeGen.generate(assignment.get());

    // Print the generated TAC
    std::cout << "Generated Three-Address Code:" << std::endl;
    codeGen.printTAC();

    return 0;
}
```

### Compilation and Execution

To compile and run the program, save the code in a single file, say `CodeGenerator.cpp`, and compile it using:

```bash
g++ -std=c++17 -o code_generator CodeGenerator.cpp
./code_generator
```

### Expected Output

You should see output similar to this, representing the three-address code generated from the AST:

```
Generated Three-Address Code:
t0 = 5 + 3
x = t0
```

### Explanation of the Code

1. **AST Structure**: The structure for the AST remains the same as in previous examples, with nodes for variable declarations, variable usage, assignments, and binary operations.

2. **TAC Structure**: The `TACInstruction` struct holds the operator and operands, along with a method to print the instruction.

3. **Code Generator**:
   - The `CodeGenerator` class has a `generate` method that traverses the AST.
   - For each `Assignment`, it generates a TAC instruction that assigns the result of the expression to the variable.
   - For `BinaryOperation`, it generates temporary variables for the operands and creates a TAC instruction for the operation.
   - The `newTemp` method generates unique temporary variable names.

4. **Example Usage**: The `main` function constructs a simple AST for the expression `int x; x = 5 + 3;` and demonstrates how to generate and print the corresponding TAC.

This example provides a foundation for generating more complex three-address code and can be expanded to handle various types of operations, control flow, and other language constructs as needed.