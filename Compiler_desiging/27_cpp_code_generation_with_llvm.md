Integrating LLVM as a backend in a C++ compiler involves generating LLVM Intermediate Representation (IR) from your high-level code and then compiling that IR to machine code (or assembly). Below is an example demonstrating how to create a simple C++ program that generates LLVM IR and compiles it to assembly using LLVM libraries.

### Steps to Generate LLVM IR and Compile to Assembly

1. **Set Up LLVM**: Make sure you have LLVM installed on your system. You can download it from the [LLVM website](https://llvm.org/). You might also want to build LLVM from source to ensure you have the latest features.

2. **Include LLVM Headers**: Your C++ code will need to include LLVM headers and link against LLVM libraries.

3. **Generate LLVM IR**: Write C++ code to create a simple LLVM module that represents your program.

4. **Compile to Assembly**: Use LLVM's JIT compilation or LLVM's code generation capabilities to output assembly code.

### Example Code

Here is a simple example of a C++ program that uses LLVM to generate IR for a function that adds two integers and then compiles it to assembly code.

```cpp
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/FunctionType.h>
#include <llvm/IR/IntegerType.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Support/Host.h>
#include <llvm/CodeGen/CodeGenAction.h>
#include <llvm/CodeGen/TargetLowering.h>
#include <llvm/IR/Verifier.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>

using namespace llvm;

int main() {
    // Initialize LLVM
    LLVMContext context;
    Module *module = new Module("simple_module", context);
    IRBuilder<> builder(context);
    
    // Create the function signature (int add(int, int))
    FunctionType *funcType = FunctionType::get(Type::getInt32Ty(context), 
                                                {Type::getInt32Ty(context), 
                                                 Type::getInt32Ty(context)}, 
                                                false);
    Function *addFunction = Function::Create(funcType, Function::ExternalLinkage, "add", module);

    // Create a basic block and set the builder's insertion point
    BasicBlock *entry = BasicBlock::Create(context, "entry", addFunction);
    builder.SetInsertPoint(entry);
    
    // Get function arguments
    auto args = addFunction->arg_begin();
    Value *a = args++;
    Value *b = args;
    
    // Create the addition instruction
    Value *sum = builder.CreateAdd(a, b, "sum");
    
    // Return the sum
    builder.CreateRet(sum);
    
    // Verify the generated function
    verifyFunction(*addFunction);
    
    // Print the generated IR
    module->print(outs(), nullptr);

    // Initialize the JIT
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    std::string error;
    TargetMachine *targetMachine = EngineBuilder(std::unique_ptr<Module>(module)).createTargetMachine();
    if (!targetMachine) {
        errs() << "Could not create target machine\n";
        return 1;
    }

    // Set the file type
    std::error_code ec;
    llvm::raw_fd_ostream dest("output.s", ec, llvm::sys::fs::OF_None);
    if (ec) {
        errs() << "Could not open file: " << ec.message() << "\n";
        return 1;
    }

    // Emit the assembly code
    llvm::legacy::PassManager pass;
    pass.add(createPrintModulePass(dest));
    pass.run(*module);

    dest.flush();
    
    std::cout << "Assembly code has been written to output.s\n";

    return 0;
}
```

### Explanation of the Code

1. **Initialize LLVM**: Set up the LLVM context and create a new module.

2. **Function Creation**: Define a function named `add` that takes two integers as parameters and returns an integer.

3. **IR Generation**: Use `IRBuilder` to create a basic block and add instructions that perform addition. The resulting LLVM IR will represent this function.

4. **Function Verification**: Verify the generated function for correctness.

5. **Output IR**: Print the generated LLVM IR to the standard output.

6. **JIT Initialization**: Prepare to compile the module into assembly code. 

7. **File Output**: Write the assembly code to a file named `output.s`.

### Compilation and Execution

To compile this code, make sure to link against the LLVM libraries. You can use a command like this:

```bash
clang++ -o llvm_codegen llvm_codegen.cpp `llvm-config --cxxflags --ldflags --libs core executionengine orcjit native`
```

After compiling, run the executable:

```bash
./llvm_codegen
```

You should see the generated LLVM IR printed on the console, and an assembly file `output.s` will be created in the current directory.

### Conclusion

This example shows how to generate LLVM IR and compile it to assembly using LLVM's APIs in C++. The process involves creating functions, basic blocks, and operations, followed by printing or outputting the final assembly code. This setup can be expanded for more complex code generation tasks in a compiler context.