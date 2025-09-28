
---

## 🧭 STRUCTURE

| Category                               | Prompt Count |
| -------------------------------------- | ------------ |
| 🔹 A. Modern C++ Basics                | 15           |
| 🔹 B. CMake & Build Systems            | 5            |
| 🔹 C. Compiler Design & Phases         | 10           |
| 🔹 D. Flex/Bison & Frontend            | 5            |
| 🔹 E. LLVM IR & Internals              | 15           |
| 🔹 F. Data Structures (CFG, SSA, etc.) | 5            |
| 🔹 G. LLVM Source & Contribution       | 10           |
| 🔹 H. Hands-On Projects                | 5            |

---

## 🔹 A. MODERN C++ BASICS (15 Prompts)

1. What is the difference between C++98 and C++11?
2. Explain RAII and how it prevents memory leaks.
3. What are smart pointers? Show code using `unique_ptr` and `shared_ptr`.
4. How does move semantics work in C++11?
5. What are lambdas? Use one in a sorting example.
6. Difference between stack and heap in C++ with examples.
7. How do templates work? Write a function template for swap.
8. What are virtual functions? How do vtables work?
9. Create a simple class with a constructor, destructor, and copy constructor.
10. Explain `std::vector`, `std::map`, and `std::unordered_map` with examples.
11. How does inheritance work in C++? Provide a base and derived class example.
12. What is the role of `const` and `constexpr` in C++?
13. What’s the difference between shallow copy and deep copy in C++?
14. How does exception handling work in C++?
15. What are function pointers and how are they used?

---

## 🔹 B. CMAKE & BUILD SYSTEMS (5 Prompts)

16. What is CMake and why is it used in large C++ projects like LLVM?
17. Create a basic `CMakeLists.txt` for compiling a Hello World C++ program.
18. How do you use `target_link_libraries` in CMake?
19. Explain the difference between static and shared libraries using CMake.
20. Build and install a sample open-source C++ project using CMake.

---

## 🔹 C. COMPILER DESIGN & PHASES (10 Prompts)

21. List and explain all phases of a compiler.
22. What is lexical analysis? Design a basic tokenizer in C++.
23. Write a simple BNF grammar for arithmetic expressions.
24. What is syntax analysis (parsing)?
25. Define and explain Abstract Syntax Tree (AST).
26. What is semantic analysis in compiler design?
27. What is Intermediate Representation (IR)? Why is it used?
28. What are the advantages of three-address code (TAC)?
29. What is code optimization? Give examples of common optimizations.
30. What is code generation? How is machine code emitted?

---

## 🔹 D. FLEX/BISON & FRONTEND PARSING (5 Prompts)

31. What is Flex and how does it tokenize input?
32. Write a simple `calc.l` file using Flex.
33. What is Bison and how does it build a parser?
34. Write a Bison grammar for arithmetic expressions.
35. Connect a Flex lexer and Bison parser into a working calculator.

---

## 🔹 E. LLVM IR & INTERNALS (15 Prompts)

36. What is LLVM IR? Show an example of IR for a simple `add` function.
37. What are the three forms of LLVM IR? (High-level, Mid-level, Low-level)
38. What is `clang` and how does it convert C++ to LLVM IR?
39. How do you compile C code to LLVM IR using `clang -S -emit-llvm`?
40. Analyze the IR output for a simple `for` loop in C.
41. What is an LLVM Pass?
42. Write a basic LLVM FunctionPass that counts instructions.
43. How does the LLVM pass pipeline work?
44. What is `opt` and how is it used to test passes?
45. What is `llc` and how does it compile IR to assembly?
46. What are intrinsics in LLVM? Give some common examples.
47. How is control flow represented in LLVM IR?
48. What is Static Single Assignment (SSA) form in LLVM?
49. How is memory represented (alloca, load, store) in LLVM IR?
50. Explore and explain the role of `Module`, `Function`, `BasicBlock`, and `Instruction` in LLVM API.

---

## 🔹 F. DATA STRUCTURES (CFG, SSA, etc.) (5 Prompts)

51. What is a Control Flow Graph (CFG)? Create a sample CFG.
52. How is SSA form maintained in LLVM?
53. Explain `phi` nodes with IR examples.
54. What is dominance in CFG and how is it computed?
55. What are live variable and dead code elimination techniques?

---

## 🔹 G. LLVM SOURCE CODE & CONTRIBUTION (10 Prompts)

56. Clone `llvm-project` and build it using CMake and Ninja.
57. Explore the `llvm/lib/IR` directory. What does it contain?
58. Understand and explain the `llvm/include/llvm/IR/Instruction.h` file.
59. Debug an existing pass using `gdb` or `lldb`.
60. What is a good first issue in LLVM? How to find one?
61. How to write unit tests for an LLVM pass using `lit` and `FileCheck`?
62. Fix a typo or documentation issue and submit a patch.
63. What is `Phabricator` in LLVM contribution?
64. How do you submit a patch via `arc diff`?
65. Read and summarize a real merged pull request to understand review culture.

---

## 🔹 H. HANDS-ON PROJECTS (5 Prompts)

66. Build a working `toy` language using the LLVM Kaleidoscope tutorial.
67. Modify the toy language to support `if-else` or `while` loops.
68. Write an LLVM Pass that counts the number of loops in a function.
69. Create a Flex + Bison frontend that generates LLVM IR using the LLVM API.
70. Write a C++ tool that reads LLVM IR and analyzes function call depth.

---
