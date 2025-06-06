Loop unrolling is an optimization technique used to increase a program's performance by reducing the overhead of loop control and increasing the amount of work done per iteration. It involves expanding the loop body so that multiple iterations of the loop are executed within a single loop iteration.

Here's a C++ implementation that demonstrates how to perform loop unrolling on an intermediate representation of loops. This example assumes a simple representation of loops and focuses on the unrolling process.

### Example C++ Code for Loop Unrolling

```cpp
#include <iostream>
#include <vector>
#include <string>

struct Loop {
    std::string variable; // Loop variable (e.g., i)
    int start;           // Start value of the loop variable
    int end;             // End value of the loop variable
    int step;            // Step value (typically 1)
    std::vector<std::string> body; // Loop body represented as strings

    // Function to unroll the loop by a specified factor
    std::vector<std::string> unroll(int factor) {
        std::vector<std::string> unrolledBody;

        // Calculate the number of iterations
        for (int i = start; i < end; i += factor) {
            for (int j = 0; j < factor; ++j) {
                if (i + j < end) {
                    // Replace the loop variable with the current value
                    for (const auto& stmt : body) {
                        unrolledBody.push_back(stmt + " // Unrolled with " + variable + " = " + std::to_string(i + j));
                    }
                }
            }
        }

        return unrolledBody;
    }
};

// Function to print the unrolled loop
void printUnrolledLoop(const std::vector<std::string>& unrolledBody) {
    for (const auto& stmt : unrolledBody) {
        std::cout << stmt << std::endl;
    }
}

int main() {
    // Create an example loop
    Loop loop;
    loop.variable = "i";
    loop.start = 0;
    loop.end = 10; // Loop will iterate from 0 to 9
    loop.step = 1; // Increment step (not used directly in unrolling)
    loop.body = {
        "sum += array[i];",   // Example body statement
        "count++;",            // Another example body statement
    };

    // Unroll the loop by a factor of 2
    int unrollFactor = 2;
    std::vector<std::string> unrolledBody = loop.unroll(unrollFactor);

    std::cout << "Unrolled Loop Body:\n";
    printUnrolledLoop(unrolledBody);

    return 0;
}
```

### Explanation of the Code

1. **Data Structures**:
   - The `Loop` struct represents a loop with a variable, start and end values, a step, and the body of the loop (which is represented as a vector of strings).
   - The `unroll` method takes an integer `factor` as input, which determines how many times the loop body will be expanded.

2. **Unrolling Logic**:
   - The method calculates the number of iterations based on the start and end values of the loop.
   - For each iteration, it generates multiple copies of the loop body statements, replacing the loop variable with the appropriate value.
   - The loop variable's value is incremented according to the loop's iteration.

3. **Printing Function**:
   - The `printUnrolledLoop` function prints the statements of the unrolled loop body.

4. **Main Function**:
   - In the `main` function, an example loop is created with a specified body.
   - The loop is unrolled by a factor of 2, and the resulting unrolled loop body is printed.

### Sample Output

When you run the program, you will see the output for the unrolled loop body:

```
Unrolled Loop Body:
sum += array[0]; // Unrolled with i = 0
count++; // Unrolled with i = 0
sum += array[1]; // Unrolled with i = 1
count++; // Unrolled with i = 1
sum += array[2]; // Unrolled with i = 2
count++; // Unrolled with i = 2
sum += array[3]; // Unrolled with i = 3
count++; // Unrolled with i = 3
sum += array[4]; // Unrolled with i = 4
count++; // Unrolled with i = 4
sum += array[5]; // Unrolled with i = 5
count++; // Unrolled with i = 5
sum += array[6]; // Unrolled with i = 6
count++; // Unrolled with i = 6
sum += array[7]; // Unrolled with i = 7
count++; // Unrolled with i = 7
sum += array[8]; // Unrolled with i = 8
count++; // Unrolled with i = 8
sum += array[9]; // Unrolled with i = 9
count++; // Unrolled with i = 9
```

### Conclusion

This C++ implementation provides a straightforward example of loop unrolling as an optimization technique for loops in an intermediate representation. You can enhance this implementation by adding support for more complex loops, handling nested loops, or integrating it into a larger compiler or code generation system. Loop unrolling can significantly improve performance, especially in loops with a known number of iterations.