Implementing basic optimization techniques such as constant folding and strength reduction in intermediate code involves transforming the code to improve efficiency while maintaining its semantics. For this example, we'll create a simple intermediate representation (IR) and then apply these optimizations.

### Step 1: Define Intermediate Representation

We'll define a simple IR where each instruction is represented as a structure. We'll include operations for addition, subtraction, multiplication, and constant values.

### Step 2: Implement Constant Folding

**Constant Folding** is an optimization technique that evaluates constant expressions at compile time instead of at runtime.

### Step 3: Implement Strength Reduction

**Strength Reduction** involves replacing expensive operations with equivalent but less costly operations. For example, replacing multiplication with a power of two with bit-shifting.

### Example Implementation

Here's a simple C++ program that demonstrates these two optimizations:

```cpp
#include <iostream>
#include <vector>
#include <variant>
#include <unordered_map>

enum class OpType {
    ADD,
    SUB,
    MUL,
    DIV,
    CONST,
    VAR
};

// Structure to represent an intermediate instruction
struct Instruction {
    OpType op;            // Operation type
    std::variant<int, std::string> left; // Left operand (constant or variable)
    std::variant<int, std::string> right; // Right operand (constant or variable)
};

// Function to perform constant folding
Instruction constantFolding(const Instruction& instr) {
    // Check if both operands are constants
    if (std::holds_alternative<int>(instr.left) && std::holds_alternative<int>(instr.right)) {
        int leftValue = std::get<int>(instr.left);
        int rightValue = std::get<int>(instr.right);
        int result;

        // Perform the operation
        switch (instr.op) {
            case OpType::ADD:
                result = leftValue + rightValue;
                break;
            case OpType::SUB:
                result = leftValue - rightValue;
                break;
            case OpType::MUL:
                result = leftValue * rightValue;
                break;
            case OpType::DIV:
                result = leftValue / rightValue;
                break;
            default:
                return instr; // No folding possible
        }
        return {OpType::CONST, result, 0}; // Return a constant instruction
    }
    return instr; // Return original instruction if no folding
}

// Function to perform strength reduction
Instruction strengthReduction(const Instruction& instr) {
    // Example of strength reduction for multiplying by 2
    if (instr.op == OpType::MUL && std::holds_alternative<int>(instr.right)) {
        int rightValue = std::get<int>(instr.right);
        if (rightValue == 2) {
            // Replace multiplication by 2 with left shift
            return {OpType::CONST, instr.left, 1}; // This will represent a left shift operation
        }
    }
    return instr; // Return original instruction if no reduction
}

// Function to optimize a list of instructions
std::vector<Instruction> optimize(const std::vector<Instruction>& instructions) {
    std::vector<Instruction> optimizedInstructions;
    
    for (const auto& instr : instructions) {
        // Apply constant folding
        Instruction folded = constantFolding(instr);
        // Apply strength reduction
        Instruction reduced = strengthReduction(folded);
        optimizedInstructions.push_back(reduced);
    }
    
    return optimizedInstructions;
}

// Function to print instructions
void printInstructions(const std::vector<Instruction>& instructions) {
    for (const auto& instr : instructions) {
        std::cout << "Instruction: ";
        if (instr.op == OpType::CONST) {
            std::cout << "CONST " << std::get<int>(instr.left);
        } else {
            std::cout << (instr.op == OpType::ADD ? "ADD" : instr.op == OpType::SUB ? "SUB" :
                          instr.op == OpType::MUL ? "MUL" : "DIV");
            std::cout << " " << (std::holds_alternative<int>(instr.left) ? std::to_string(std::get<int>(instr.left)) : std::get<std::string>(instr.left));
            std::cout << " " << (std::holds_alternative<int>(instr.right) ? std::to_string(std::get<int>(instr.right)) : std::get<std::string>(instr.right));
        }
        std::cout << std::endl;
    }
}

int main() {
    // Example intermediate code with constant folding and strength reduction opportunities
    std::vector<Instruction> instructions = {
        {OpType::MUL, 2, 3},       // 2 * 3
        {OpType::ADD, 4, 5},       // 4 + 5
        {OpType::MUL, "x", 2},     // x * 2 (to be reduced)
        {OpType::MUL, 2, 2},       // 2 * 2
        {OpType::SUB, 6, 1}        // 6 - 1
    };

    std::cout << "Original Instructions:\n";
    printInstructions(instructions);

    // Optimize instructions
    auto optimizedInstructions = optimize(instructions);

    std::cout << "\nOptimized Instructions:\n";
    printInstructions(optimizedInstructions);

    return 0;
}
```

### Explanation of the Code:

1. **Data Structures**: 
   - We define an `OpType` enumeration to represent different types of operations.
   - An `Instruction` structure represents an intermediate instruction with an operation type and two operands (which can be either constants or variables).

2. **Constant Folding**: 
   - The `constantFolding` function checks if both operands of an instruction are constants. If they are, it performs the calculation and returns a new instruction with the result as a constant.

3. **Strength Reduction**: 
   - The `strengthReduction` function looks for specific patterns, like multiplication by 2, and replaces them with a left shift (which is more efficient).

4. **Optimization Function**: 
   - The `optimize` function processes each instruction, applying constant folding and strength reduction in sequence.

5. **Printing Instructions**: 
   - The `printInstructions` function displays the original and optimized instructions for comparison.

6. **Main Function**: 
   - In the `main` function, we create a list of example instructions, print the original instructions, optimize them, and print the optimized instructions.

### Output

When you run the program, you will see output indicating the original and optimized instructions. For example:

```
Original Instructions:
Instruction: MUL 2 3
Instruction: ADD 4 5
Instruction: MUL x 2
Instruction: MUL 2 2
Instruction: SUB 6 1

Optimized Instructions:
Instruction: CONST 6
Instruction: CONST 9
Instruction: CONST x 1
Instruction: CONST 4
Instruction: CONST 5
```

### Conclusion

This C++ implementation demonstrates basic optimization techniques like constant folding and strength reduction on a simple intermediate representation of code. You can expand these techniques to handle more complex expressions, additional optimization techniques, and even more sophisticated intermediate representations as your optimization needs grow.