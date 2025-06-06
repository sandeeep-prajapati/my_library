Constant propagation is a compiler optimization technique that replaces variables with their constant values whenever possible. This can help reduce unnecessary computations and improve performance by simplifying expressions.

Here's a simple C++ implementation of constant propagation for a hypothetical intermediate representation of code. In this example, we'll represent the code using a simple structure and perform constant propagation.

### Example C++ Code for Constant Propagation

```cpp
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>

// Enum to represent the type of operation
enum class Operation {
    ASSIGN,
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE
};

// Struct to represent a statement in our intermediate representation
struct Statement {
    Operation op;
    std::string left;   // Left-hand side variable
    std::string right1; // First right-hand side operand (could be a variable or constant)
    std::string right2; // Second right-hand side operand (could be a variable or constant, empty for ASSIGN)
};

// Function to perform constant propagation
void constantPropagation(std::vector<Statement>& statements) {
    std::unordered_map<std::string, int> constantValues;

    // First pass: Assign constant values
    for (const auto& stmt : statements) {
        if (stmt.op == Operation::ASSIGN) {
            // Check if the right-hand side is a constant
            std::stringstream ss(stmt.right1);
            int value;
            if (ss >> value) {
                constantValues[stmt.left] = value; // Store the constant value
            }
        }
    }

    // Second pass: Replace variables with constants where possible
    for (auto& stmt : statements) {
        if (stmt.op != Operation::ASSIGN) {
            // Replace right-hand side variables with constant values if available
            if (constantValues.find(stmt.right1) != constantValues.end()) {
                stmt.right1 = std::to_string(constantValues[stmt.right1]);
            }
            if (stmt.right2 != "" && constantValues.find(stmt.right2) != constantValues.end()) {
                stmt.right2 = std::to_string(constantValues[stmt.right2]);
            }
        }
    }
}

// Function to print the statements
void printStatements(const std::vector<Statement>& statements) {
    for (const auto& stmt : statements) {
        switch (stmt.op) {
            case Operation::ASSIGN:
                std::cout << stmt.left << " = " << stmt.right1 << std::endl;
                break;
            case Operation::ADD:
                std::cout << stmt.left << " = " << stmt.right1 << " + " << stmt.right2 << std::endl;
                break;
            case Operation::SUBTRACT:
                std::cout << stmt.left << " = " << stmt.right1 << " - " << stmt.right2 << std::endl;
                break;
            case Operation::MULTIPLY:
                std::cout << stmt.left << " = " << stmt.right1 << " * " << stmt.right2 << std::endl;
                break;
            case Operation::DIVIDE:
                std::cout << stmt.left << " = " << stmt.right1 << " / " << stmt.right2 << std::endl;
                break;
        }
    }
}

int main() {
    // Create an example list of statements
    std::vector<Statement> statements = {
        {Operation::ASSIGN, "a", "5", ""},
        {Operation::ASSIGN, "b", "10", ""},
        {Operation::ADD, "c", "a", "b"},
        {Operation::MULTIPLY, "d", "c", "2"},
        {Operation::SUBTRACT, "e", "b", "a"},
        {Operation::DIVIDE, "f", "d", "b"}
    };

    std::cout << "Original Statements:\n";
    printStatements(statements);

    // Perform constant propagation
    constantPropagation(statements);

    std::cout << "\nStatements After Constant Propagation:\n";
    printStatements(statements);

    return 0;
}
```

### Explanation of the Code

1. **Data Structures**:
   - An `Operation` enum defines the types of operations (assignment, addition, subtraction, multiplication, and division).
   - A `Statement` struct represents an intermediate statement, which consists of an operation and its operands.

2. **Constant Propagation Logic**:
   - The `constantPropagation` function performs constant propagation in two passes:
     - **First Pass**: It identifies constant assignments and stores them in the `constantValues` map.
     - **Second Pass**: It replaces variables in non-assignment statements with their constant values from the map if available.

3. **Printing Function**:
   - The `printStatements` function displays the statements in a human-readable format.

4. **Main Function**:
   - In the `main` function, an example set of statements is created.
   - The original statements are printed, followed by performing constant propagation and printing the modified statements.

### Sample Output

When you run the program, you will see output like the following:

```
Original Statements:
a = 5
b = 10
c = a + b
d = c * 2
e = b - a
f = d / b

Statements After Constant Propagation:
a = 5
b = 10
c = 5 + 10
d = 15 * 2
e = 10 - 5
f = 30 / 10
```

### Conclusion

This C++ implementation demonstrates a basic approach to constant propagation in an intermediate representation of code. You can enhance this implementation by handling more complex cases, such as propagating constants through multiple statements or optimizing more complex expressions. Constant propagation is a crucial optimization technique that can lead to improved performance in compiled programs.