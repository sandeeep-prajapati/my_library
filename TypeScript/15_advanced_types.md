### **Advanced Types in TypeScript**

TypeScript offers advanced types that allow for more flexible and dynamic type definitions. These advanced types can help in creating highly reusable, generic, and type-safe code. Some of the key advanced types in TypeScript include **Mapped Types**, **Conditional Types**, and **Template Literal Types**. Let’s explore each of these types and how they enhance the flexibility of the language.

---

### **1. Mapped Types**

A **Mapped Type** allows you to create new types by transforming the properties of an existing type. This transformation can include actions like adding, removing, or modifying properties. Mapped types are particularly useful for generating a type based on another type, providing flexibility in how types can be derived.

#### **Syntax:**
```typescript
type MappedType<T> = {
  [K in keyof T]: T[K];
};
```

- `keyof T` is used to get all the keys of the type `T`.
- `[K in keyof T]` iterates over each key in `T`.
- The resulting type is a new type where the properties of `T` are transformed according to the defined logic.

#### **Example of a Mapped Type:**
```typescript
type Person = {
  name: string;
  age: number;
};

type ReadOnlyPerson = {
  readonly [K in keyof Person]: Person[K];
};

const person: ReadOnlyPerson = {
  name: "John",
  age: 30,
};

person.name = "Jane"; // Error: cannot assign to 'name' because it is a read-only property.
```

In this example:
- `ReadOnlyPerson` is a **Mapped Type** that makes all properties of `Person` read-only by using the `readonly` modifier.

#### **Use Cases:**
- Making properties of an object read-only.
- Creating a type with optional properties.
- Dynamically generating a type based on existing properties.

---

### **2. Conditional Types**

**Conditional Types** allow you to define types based on a condition, much like an `if-else` statement. This conditional type checks whether a type extends another type and returns different types based on that condition. They enable highly flexible type transformations based on type relationships.

#### **Syntax:**
```typescript
T extends U ? X : Y;
```

- If `T` extends `U`, the type resolves to `X`; otherwise, it resolves to `Y`.

#### **Example of a Conditional Type:**
```typescript
type IsString<T> = T extends string ? "Yes" : "No";

type Test1 = IsString<string>; // "Yes"
type Test2 = IsString<number>; // "No"
```

In this example:
- The `IsString<T>` type uses a **Conditional Type** to check if `T` is a `string`. If true, it returns `"Yes"`; otherwise, it returns `"No"`.

#### **Use Cases:**
- Conditionally determining types based on input.
- Building type transformations that depend on type conditions.
- Customizing return types in functions based on input types.

---

### **3. Template Literal Types**

**Template Literal Types** allow you to create types by embedding expressions inside string templates. This provides a powerful way to define string-based types that can combine literal strings with variables or other types. They offer a more dynamic way of creating strings that are type-safe.

#### **Syntax:**
```typescript
type MyString = `Hello, ${string}!`;
```

- This type defines a string that must start with `"Hello, "` and end with `"!"`, with any `string` in between.

#### **Example of Template Literal Types:**
```typescript
type Direction = "left" | "right";
type MoveCommand = `move ${Direction}`;

let move: MoveCommand = "move left"; // Valid
move = "move up"; // Error: Type '"move up"' is not assignable to type 'MoveCommand'.
```

In this example:
- `MoveCommand` is a **Template Literal Type** that ensures the string starts with `"move "` and is followed by one of the possible directions (`"left"` or `"right"`).

#### **Use Cases:**
- Creating flexible string-based types that combine literal values with dynamic content.
- Enforcing string patterns for function names, URLs, paths, or commands.
- Defining complex string formats like paths, IDs, or query strings.

---

### **4. Combination of Advanced Types**

TypeScript allows combining **Mapped Types**, **Conditional Types**, and **Template Literal Types** to create complex, dynamic types that offer strong type-safety guarantees while allowing flexibility in handling different scenarios.

#### **Example Combining Multiple Advanced Types:**
```typescript
type MyObject = {
  id: number;
  name: string;
};

type OptionalFields<T> = {
  [K in keyof T]?: T[K];  // Makes all fields optional
};

type PrefixedKeys<T> = {
  [K in keyof T as `prefix_${string & K}`]: T[K];  // Adds prefix to all keys
};

type FlexibleType = OptionalFields<MyObject>;
type PrefixedType = PrefixedKeys<MyObject>;

const obj1: FlexibleType = { id: 10 }; // Optional fields
const obj2: PrefixedType = { prefix_id: 10, prefix_name: "John" }; // Prefixed keys
```

In this example:
- **Mapped Type** (`OptionalFields`) makes all properties optional.
- **Mapped Type with Template Literal** (`PrefixedKeys`) adds a prefix to all keys in the type.

---

### **Conclusion**

- **Mapped Types** allow the transformation of existing types, making them highly flexible for tasks like modifying properties, creating partial types, or adding/removing properties dynamically.
- **Conditional Types** provide type transformations based on conditions, enabling more dynamic and conditional behavior in type definitions.
- **Template Literal Types** allow creating string-based types with embedded expressions, offering powerful string handling capabilities and ensuring the types conform to specific patterns.

By leveraging these advanced types, TypeScript improves its expressiveness and flexibility, allowing developers to write more generic, reusable, and type-safe code.