In TypeScript, data types help define the kind of values variables can hold. TypeScript’s static typing ensures that values assigned to variables match their expected types, which helps catch errors during development rather than at runtime.

### Different Data Types in TypeScript:

1. **Primitive Types:**
   These are the basic building blocks of any TypeScript program.

   - **`number`:** Represents both integer and floating-point numbers.
     ```typescript
     let age: number = 25;
     let price: number = 19.99;
     ```

   - **`string`:** Represents a sequence of characters (text).
     ```typescript
     let name: string = "Alice";
     let message: string = `Hello, ${name}!`;
     ```

   - **`boolean`:** Represents a value that is either `true` or `false`.
     ```typescript
     let isActive: boolean = true;
     let hasPermission: boolean = false;
     ```

   - **`null`:** Represents the intentional absence of any value. The value `null` itself is the only possible value of type `null`.
     ```typescript
     let selectedItem: null = null;
     ```

   - **`undefined`:** Represents an uninitialized variable or an absence of a value. The value `undefined` is the only possible value of type `undefined`.
     ```typescript
     let userInfo: undefined = undefined;
     ```

2. **Object Types:**
   These types represent instances of objects, including arrays and functions.

   - **`object`:** A non-primitive type that can represent any value that is not a primitive (e.g., arrays, functions, or custom objects).
     ```typescript
     let person: object = { name: "Bob", age: 30 };
     ```

   - **`Array`:** Represents a list of values of a specific type.
     ```typescript
     let numbers: number[] = [1, 2, 3];
     let names: string[] = ["Alice", "Bob"];
     ```

   - **`tuple`:** Represents an ordered collection of elements of fixed sizes and types.
     ```typescript
     let employee: [string, number] = ["John", 25];
     ```

   - **`enum`:** A special type used to define a set of named constants.
     ```typescript
     enum Direction {
       Up = 1,
       Down,
       Left,
       Right
     }
     let move: Direction = Direction.Up;
     ```

3. **Function Types:**
   TypeScript allows you to define types for functions as well.

   - **Function Type:** Specifies the type of a function, including its parameters and return type.
     ```typescript
     let greet: (name: string) => string = (name) => `Hello, ${name}`;
     ```

4. **`any` Type:**
   Represents any value, similar to JavaScript. It allows a variable to hold any type of value and disables type checking for that variable.
   ```typescript
   let randomValue: any = 5;
   randomValue = "Hello";
   randomValue = true;
   ```

5. **`unknown` Type:**
   The `unknown` type is similar to `any`, but it is safer because you cannot perform any operations on a variable of type `unknown` until you narrow its type with a type guard.
   ```typescript
   let uncertainValue: unknown = 10;
   if (typeof uncertainValue === "string") {
     console.log(uncertainValue.length); // Safe to access `.length`
   }
   ```

6. **`never` Type:**
   Represents values that will never occur, such as a function that always throws an error or has an infinite loop.
   ```typescript
   function throwError(message: string): never {
     throw new Error(message);
   }
   ```

7. **`void` Type:**
   Represents the absence of a return value. Often used for functions that do not return a value.
   ```typescript
   function logMessage(message: string): void {
     console.log(message);
   }
   ```

### How Static Typing Helps in Reducing Runtime Errors:

Static typing refers to the practice of defining variable types at compile time, and it plays a crucial role in reducing runtime errors. Here’s how TypeScript’s static typing improves code safety:

1. **Early Error Detection:**
   - With static typing, TypeScript checks types during the compilation process. If you attempt to assign a variable with a value that doesn’t match its defined type, TypeScript will generate an error at compile time, preventing the code from running with potential type-related issues.
   - Example:
     ```typescript
     let age: number = 25;
     age = "twenty-five"; // Error: Type 'string' is not assignable to type 'number'.
     ```

2. **Type Checking in Functions:**
   - TypeScript ensures that function parameters and return types match the expected types. If you call a function with arguments that do not align with its expected types, TypeScript will raise an error.
   - Example:
     ```typescript
     function greet(name: string): string {
       return `Hello, ${name}!`;
     }
     greet(42); // Error: Argument of type 'number' is not assignable to parameter of type 'string'.
     ```

3. **Better Code Documentation and Readability:**
   - The use of explicit types serves as **self-documentation** for your code. This makes it easier for other developers (or even yourself) to understand what type of data a function or variable is expected to work with, leading to fewer misunderstandings and mistakes.
   - Example:
     ```typescript
     let price: number = 19.99; // It is clear that 'price' should always be a number.
     ```

4. **Preventing Common Errors:**
   - Static typing helps catch common errors, such as calling methods or accessing properties on `null` or `undefined` objects, or passing incorrect arguments to functions.
   - Example:
     ```typescript
     let user: { name: string; age: number } | null = null;
     console.log(user.name); // Error: Object is possibly 'null'.
     ```

5. **Enhanced IDE Support:**
   - TypeScript provides better **autocompletion, IntelliSense**, and **type checking** in modern IDEs, which helps developers write correct code faster. These features reduce the likelihood of introducing errors because the IDE can suggest valid operations based on variable types.
   
6. **Refactoring Confidence:**
   - When making changes or refactoring large codebases, static typing helps ensure that you are not inadvertently introducing type-related bugs. The TypeScript compiler will alert you if the changes break any type constraints, ensuring safer modifications.

7. **Consistency Across the Codebase:**
   - By enforcing types throughout the codebase, static typing ensures that the code is **consistent** and behaves as expected. This makes the code more predictable and reduces the chances of runtime errors, especially in large projects with multiple contributors.

### Conclusion:

TypeScript’s static typing provides several advantages over JavaScript, such as early error detection, better code documentation, and improved IDE support. By enforcing type constraints at compile time, TypeScript helps developers catch errors before running the application, leading to fewer runtime errors and more maintainable, robust code.