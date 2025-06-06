Setting up a C/C++ project for building a basic compiler can be done using either CMake or a Makefile. Here’s a step-by-step guide for both methods:

### Using CMake

CMake is a cross-platform tool that automates the build process for software projects. Here's how to set up a basic compiler project using CMake:

#### Step 1: Install CMake

Make sure you have CMake installed. You can download it from the [CMake website](https://cmake.org/download/).

#### Step 2: Project Structure

Create a directory structure for your compiler project:

```
my_compiler/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── lexer.cpp
│   ├── parser.cpp
├── include/
│   ├── lexer.h
│   ├── parser.h
└── build/
```

#### Step 3: Write CMakeLists.txt

In the `CMakeLists.txt` file, define your project and specify the source files:

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyCompiler)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)

# Source files
file(GLOB SOURCES "src/*.cpp")

# Create the executable
add_executable(my_compiler ${SOURCES})
```

#### Step 4: Write Source Files

You can write your basic compiler components in the `src` directory (like `lexer.cpp`, `parser.cpp`, and `main.cpp`). For example, here’s a simple `main.cpp`:

```cpp
#include <iostream>

int main() {
    std::cout << "Hello, Compiler!" << std::endl;
    return 0;
}
```

#### Step 5: Build the Project

1. Open a terminal and navigate to the project root directory (`my_compiler`).
2. Create the build directory and navigate into it:
   ```bash
   mkdir build
   cd build
   ```
3. Run CMake to generate the build files:
   ```bash
   cmake ..
   ```
4. Build the project:
   ```bash
   make
   ```

Your compiler executable (`my_compiler`) will be created in the `build` directory.

### Using Makefile

If you prefer to use a Makefile, follow these steps:

#### Step 1: Project Structure

Use the same project structure as mentioned above.

#### Step 2: Write the Makefile

Create a `Makefile` in the project root directory:

```makefile
CXX = g++
CXXFLAGS = -std=c++11 -Iinclude

# Source files
SOURCES = src/main.cpp src/lexer.cpp src/parser.cpp
# Executable name
EXECUTABLE = my_compiler

# Build target
all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Clean target
clean:
	rm -f $(EXECUTABLE)
```

#### Step 3: Write Source Files

Similar to the CMake setup, write your source files in the `src` directory.

#### Step 4: Build the Project

1. Open a terminal and navigate to the project root directory (`my_compiler`).
2. Run the following command to build the project:
   ```bash
   make
   ```
3. To clean the build artifacts, run:
   ```bash
   make clean
   ```

Your compiler executable (`my_compiler`) will be created in the project root directory.

### Summary

Both CMake and Makefile provide a straightforward way to set up and build a C/C++ project for a basic compiler. CMake is more versatile and works across platforms, while Makefiles are simple and effective for straightforward projects. Choose the method that best fits your development workflow!