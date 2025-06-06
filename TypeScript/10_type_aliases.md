### **Type Aliases in TypeScript**

A **type alias** in TypeScript allows you to create a new name for a type, which can be any valid TypeScript type (primitive types, object types, function types, union types, etc.). Type aliases are defined using the `type` keyword and are often used to simplify complex type definitions or to provide a more meaningful name for a type.

#### **Syntax of Type Alias:**

```typescript
type TypeName = TypeDefinition;
```

For example:

```typescript
type StringOrNumber = string | number;
```

In this example, `StringOrNumber` is a type alias for the union type `string | number`.

---

### **How Type Aliases Work:**

Type aliases work by creating a new name for a type that can be used wherever the type is needed. They do not create new types, but rather provide a shorthand for existing types or combinations of types. You can use them to define primitive types, object types, function types, union types, and intersection types.

#### **Examples:**

1. **Primitive Type Alias:**

```typescript
type StringAlias = string;

let message: StringAlias = "Hello, TypeScript!";
```

Here, `StringAlias` is just an alias for the `string` type, but it can be useful to convey the purpose of the variable.

2. **Union Type Alias:**

```typescript
type StringOrBoolean = string | boolean;

let value: StringOrBoolean = "Hello";  // Valid
value = true;                         // Valid
value = 42;                           // Error
```

3. **Object Type Alias:**

```typescript
type Point = { x: number; y: number };

let point: Point = { x: 10, y: 20 };
```

4. **Function Type Alias:**

```typescript
type Operation = (a: number, b: number) => number;

const add: Operation = (a, b) => a + b;
```

5. **Intersection Type Alias:**

```typescript
type Shape = { color: string };
type Circle = { radius: number };

type ColoredCircle = Shape & Circle;

let circle: ColoredCircle = { color: "red", radius: 10 };
```

---

### **When to Use Type Aliases:**

Type aliases are helpful in several scenarios:

1. **Complex Type Definitions:**
   When a type definition is complex or repetitive, using a type alias can simplify your code.

   ```typescript
   type StringOrNumber = string | number;
   type Coordinate = { x: number, y: number };
   type Callback = (result: boolean) => void;
   ```

   Without aliases, these would need to be repeated in your code, which makes it harder to maintain.

2. **Creating Meaningful Names for Types:**
   Type aliases allow you to create more descriptive names for types, improving code readability.

   ```typescript
   type ID = string | number;
   ```

   Here, `ID` is a more meaningful name than `string | number` for variables that represent identifiers.

3. **Union and Intersection Types:**
   Type aliases are particularly useful when dealing with union (`|`) or intersection (`&`) types, making the code more readable and reusable.

   ```typescript
   type Shape = { color: string };
   type Circle = Shape & { radius: number };
   ```

   Instead of writing complex intersections directly, aliases make them easier to understand and maintain.

4. **Function Signatures:**
   If you have functions that use the same signature in multiple places, creating a type alias can prevent repetition.

   ```typescript
   type Calculate = (x: number, y: number) => number;

   const add: Calculate = (x, y) => x + y;
   const subtract: Calculate = (x, y) => x - y;
   ```

---

### **When to Use Type Aliases vs Interfaces:**

In TypeScript, both **type aliases** and **interfaces** are used to define custom types, but they have some key differences and use cases. Here's a comparison:

| Feature                 | **Type Alias**                                   | **Interface**                                     |
|-------------------------|--------------------------------------------------|---------------------------------------------------|
| **Declaration Merging** | No, type aliases cannot be merged.              | Yes, interfaces can be merged using declaration merging. |
| **Extending Other Types** | Can extend with union types or intersections.    | Can extend other interfaces using `extends`.      |
| **Use for Objects**     | Can define object shapes but with no merging.    | Best suited for object shapes and extending objects. |
| **Use for Functions**   | Ideal for function signatures and type unions.  | Can be used for functions but not as flexible as type aliases. |
| **Best for**            | More flexible (works for any type).             | Better for defining object-oriented structures.   |

#### **Use Cases for Type Aliases:**

1. **Unions and Intersections:**
   Type aliases are often preferred when you need to combine multiple types using unions (`|`) or intersections (`&`).

   ```typescript
   type Admin = { role: "admin" };
   type User = { role: "user" };
   type Role = Admin | User;
   ```

2. **Complex Types:**
   Type aliases are ideal for complex function signatures or when you want to use a shorthand for types that involve unions, intersections, or other complex structures.

   ```typescript
   type Validator = (value: string | number) => boolean;
   ```

3. **Primitive Aliases:**
   While you can create type aliases for simple types, they may be less common than other use cases. However, they can still improve readability.

   ```typescript
   type ID = string;
   ```

---

### **When Not to Use Type Aliases:**

- **When you need object shape inheritance**: If you need to extend the object shape or want to merge multiple declarations, **interfaces** are more appropriate.
  
- **When declaration merging is required**: Interfaces support declaration merging, whereas type aliases do not. If you expect to need this feature, interfaces are a better choice.

  ```typescript
  interface Person {
    name: string;
  }

  interface Person {
    age: number;
  }

  const person: Person = { name: "John", age: 30 };
  ```

---

### **Conclusion:**

- **Use Type Aliases** when you need:
  - Union or intersection types.
  - A more flexible and concise way to define complex types.
  - To define function signatures or primitives with custom names.
  
- **Use Interfaces** when you need:
  - A way to define object shapes that may be extended or merged.
  - To take advantage of object-oriented programming principles (like inheritance).

Both type aliases and interfaces serve different but complementary purposes, and understanding when to use each is key to writing clean, maintainable TypeScript code.