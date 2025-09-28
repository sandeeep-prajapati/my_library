
---

## ✅ 1. **LLVM ([https://github.com/llvm/llvm-project](https://github.com/llvm/llvm-project))**

### 🧠 What You Should Know:

#### 🛠️ Skills & Tools:

| Concept                                | Why It's Needed                                     |
| -------------------------------------- | --------------------------------------------------- |
| **C++ (Modern)**                       | Core language used across LLVM codebase             |
| **CMake**                              | Build system to compile LLVM and its tools          |
| **Compiler Phases**                    | Know how Lexing, Parsing, AST, IR, and CodeGen work |
| **Intermediate Representation (IR)**   | LLVM’s core (LLVM IR) – you must understand it      |
| **Bison / Flex (Yacc/Lex-like tools)** | Optional, but helps for frontend contributions      |
| **Data Structures (CFG, SSA, etc.)**   | Key for optimization and analysis passes            |

### 📚 Topics to Learn Before Contributing:

* What is LLVM IR and how it's structured
* How `clang` (LLVM's C/C++ frontend) works
* Basic C++ concepts like RAII, templates, smart pointers
* Building and debugging large C++ projects

### 🔗 Resources:

* [LLVM Getting Started Guide](https://llvm.org/docs/GettingStarted.html)
* [LLVM Tutorial: Kaleidoscope](https://llvm.org/docs/tutorial/index.html)
* YouTube: *"Building your first LLVM pass"*

---

## ✅ 2. **GraalVM ([https://github.com/oracle/graal](https://github.com/oracle/graal))**

### 🧠 What You Should Know:

#### 🛠️ Skills & Tools:

| Concept                               | Why It's Needed                                 |
| ------------------------------------- | ----------------------------------------------- |
| **Java (OOP, Generics, Annotations)** | GraalVM and Truffle framework are in Java       |
| **Truffle Framework**                 | For writing interpreters for new languages      |
| **Abstract Syntax Trees (AST)**       | Core part of GraalVM interpreter/compiler model |
| **Parser + Compiler Design**          | How bytecode and IR are created from source     |
| **JVM Internals**                     | Understanding Java runtime, JIT, etc.           |

### 📚 Topics to Learn Before Contributing:

* Java basics (Collections, Streams, Annotations)
* Truffle DSL and how Graal generates code
* What Polyglot execution means (support for JS, Python, Ruby, etc.)
* Structure of the `compiler/` folder (e.g., `graal-java`, `graal-nodejs`)

### 🔗 Resources:

* [GraalVM docs](https://www.graalvm.org/)
* [Truffle framework overview](https://www.graalvm.org/truffle/)
* Example: [SimpleLanguage – a sample language for GraalVM](https://github.com/graalvm/simplelanguage)

---

## ✅ 3. **Tree-sitter ([https://github.com/tree-sitter/tree-sitter](https://github.com/tree-sitter/tree-sitter))**

### 🧠 What You Should Know:

#### 🛠️ Skills & Tools:

| Concept                               | Why It's Needed                                          |
| ------------------------------------- | -------------------------------------------------------- |
| **C Programming (pointers, structs)** | Core Tree-sitter engine is in C                          |
| **JavaScript (Node.js)**              | Used for testing grammars and language setup             |
| **Bison-style Grammar DSL**           | Tree-sitter grammars look like simplified Bison grammars |
| **Parsing Expression Grammar (PEG)**  | Tree-sitter is based on PEG, not CFG                     |
| **AST Construction**                  | Building syntax trees from parsing rules                 |

### 📚 Topics to Learn Before Contributing:

* How to define a grammar in Tree-sitter (`grammar.js`)
* How parsing and incremental parsing works
* How Tree-sitter handles different languages (check `tree-sitter-python`, `tree-sitter-cpp`)
* How syntax highlighting and error detection are done

### 🔗 Resources:

* [Tree-sitter Tutorial](https://tree-sitter.github.io/tree-sitter/)
* Explore: [`tree-sitter-python`](https://github.com/tree-sitter/tree-sitter-python)
* YouTube: *"Building a grammar with Tree-sitter"*

---

## 🧪 Summary Table: What You Need to Learn

| Project         | Language(s) | Key Concepts to Learn              |
| --------------- | ----------- | ---------------------------------- |
| **LLVM**        | C++, CMake  | Compiler Phases, LLVM IR, CFG, SSA |
| **GraalVM**     | Java        | Truffle, ASTs, Polyglot Runtime    |
| **Tree-sitter** | C, JS       | PEG Grammar, ASTs, Grammar DSL     |

---
