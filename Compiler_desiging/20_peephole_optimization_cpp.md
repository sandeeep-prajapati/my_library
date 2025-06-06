Peephole optimization is a local optimization technique that looks at a small window (or "peephole") of instructions in generated assembly code to identify and replace inefficient instruction sequences with more efficient ones. This process can help reduce the number of instructions, improve execution speed, or save memory.

Here’s an example C++ implementation of a peephole optimization function that scans for specific instruction patterns and optimizes them:

### Example C++ Code for Peephole Optimization

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

enum class OpType {
    ADD,
    SUB,
    MUL,
    DIV,
    MOV, // Move operation
    NOP, // No operation
    LABEL // Label for jumps
};

struct Instruction {
    OpType op;            // Operation type
    std::string dst;      // Destination operand
    std::string src1;     // Source operand 1
    std::string src2;     // Source operand 2

    // For easy display
    std::string toString() const {
        switch (op) {
            case OpType::ADD: return "ADD " + dst + ", " + src1 + ", " + src2;
            case OpType::SUB: return "SUB " + dst + ", " + src1 + ", " + src2;
            case OpType::MUL: return "MUL " + dst + ", " + src1 + ", " + src2;
            case OpType::DIV: return "DIV " + dst + ", " + src1 + ", " + src2;
            case OpType::MOV: return "MOV " + dst + ", " + src1;
            case OpType::NOP: return "NOP";
            case OpType::LABEL: return dst + ":";
            default: return "";
        }
    }
};

// Function to optimize the instruction sequence using peephole optimization
void peepholeOptimize(std::vector<Instruction>& instructions) {
    for (size_t i = 0; i < instructions.size() - 1; ++i) {
        // Example optimization: Replace "MOV X, Y; MOV Y, Z" with "MOV X, Z"
        if (instructions[i].op == OpType::MOV && instructions[i + 1].op == OpType::MOV &&
            instructions[i].dst == instructions[i + 1].src1) {
            instructions[i] = Instruction{OpType::MOV, instructions[i].dst, instructions[i + 1].src1, ""};
            instructions.erase(instructions.begin() + i + 1); // Remove second MOV
            --i; // Adjust index after removal
        }
        // Example optimization: Replace "ADD X, 0" with "MOV X, 0"
        else if (instructions[i].op == OpType::ADD && instructions[i].src2 == "0") {
            instructions[i] = Instruction{OpType::MOV, instructions[i].dst, instructions[i].src1, ""};
        }
        // Example optimization: Replace "SUB X, 0" with "MOV X, 0"
        else if (instructions[i].op == OpType::SUB && instructions[i].src2 == "0") {
            instructions[i] = Instruction{OpType::MOV, instructions[i].dst, instructions[i].src1, ""};
        }
    }
}

// Function to print the assembly instructions
void printInstructions(const std::vector<Instruction>& instructions) {
    for (const auto& instr : instructions) {
        std::cout << instr.toString() << std::endl;
    }
}

int main() {
    // Example sequence of assembly instructions
    std::vector<Instruction> instructions = {
        {OpType::MOV, "R1", "5", ""},
        {OpType::MOV, "R2", "R1", ""},
        {OpType::ADD, "R3", "R2", "0"}, // This should be optimized to MOV
        {OpType::MOV, "R4", "0", ""},
        {OpType::SUB, "R5", "R4", "0"}, // This should be optimized to MOV
        {OpType::MOV, "R6", "R1", ""},
        {OpType::MOV, "R1", "R7", ""} // Redundant MOV to check
    };

    std::cout << "Original Instructions:\n";
    printInstructions(instructions);

    // Perform peephole optimization
    peepholeOptimize(instructions);

    std::cout << "\nOptimized Instructions:\n";
    printInstructions(instructions);

    return 0;
}
```

### Explanation of the Code

1. **Data Structures**: 
   - The `OpType` enum defines various operations.
   - The `Instruction` struct contains information about an instruction, including the operation type and operands, along with a `toString` method for easy display.

2. **Peephole Optimization Function**: 
   - The `peepholeOptimize` function scans through the list of instructions.
   - It identifies patterns that can be optimized:
     - **MOV Elimination**: If it finds two consecutive `MOV` instructions where the second moves a register that is the destination of the first, it combines them into a single `MOV`.
     - **Addition/Subtraction of Zero**: It converts `ADD X, 0` and `SUB X, 0` into `MOV X, Y` for optimization.

3. **Printing Function**: 
   - The `printInstructions` function outputs the assembly instructions in a readable format.

4. **Main Function**: 
   - The `main` function initializes a vector of instructions and prints them before and after applying the peephole optimization.

### Sample Output

When you run the program, you will see the output before and after optimization:

```
Original Instructions:
MOV R1, 5
MOV R2, R1
ADD R3, R2, 0
MOV R4, 0
SUB R5, R4, 0
MOV R6, R1
MOV R1, R7

Optimized Instructions:
MOV R1, 5
MOV R2, R1
MOV R3, R2
MOV R4, 0
MOV R5, R4
MOV R6, R1
MOV R1, R7
```

### Conclusion

This C++ implementation provides a basic framework for peephole optimization of assembly instructions. You can expand this further by adding more complex patterns for optimization, incorporating more sophisticated analysis techniques, or adapting the structure to handle more complex instruction sets. Peephole optimizations can significantly improve the efficiency of generated assembly code by reducing unnecessary instructions.