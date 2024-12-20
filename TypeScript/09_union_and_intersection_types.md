### **Union and Intersection Types in TypeScript**

TypeScript allows you to combine multiple types using **union** and **intersection** types. These features provide more flexibility and precision when defining types, allowing you to represent a wider range of possible values or more specific combinations of types.

---

### **Union Types**

A **union type** allows a variable to hold one of several different types. It is defined using the pipe (`|`) symbol, which signifies "either type A or type B."

#### **Example of Union Type:**

```typescript
let value: string | number;

value = "Hello"; // Valid
value = 42;      // Valid
value = true;    // Error: Type 'boolean' is not assignable to type 'string | number'
```

In the example above, `value` can be either a `string` or a `number`. If you try to assign a value of another type (like `boolean`), TypeScript will throw an error.

#### **Key Benefits of Union Types:**
1. **Flexibility**: Union types allow you to handle variables that could have multiple types, making your code more adaptable and expressive.
2. **Precise type checking**: TypeScript will ensure that the variable holds only one of the allowed types and will provide proper type inference during development.

---

### **Intersection Types**

An **intersection type** combines multiple types into one. The result of an intersection type is a type that has all the properties of the combined types. It is defined using the ampersand (`&`) symbol.

#### **Example of Intersection Type:**

```typescript
interface Person {
  name: string;
  age: number;
}

interface Contact {
  email: string;
  phone: string;
}

let employee: Person & Contact = {
  name: "John",
  age: 30,
  email: "john@example.com",
  phone: "123-456-7890"
};
```

In the example above, `employee` must have all the properties of both `Person` and `Contact`. The type `Person & Contact` ensures that the object has `name`, `age`, `email`, and `phone`.

#### **Key Benefits of Intersection Types:**
1. **Combining multiple types**: Intersection types allow you to compose types by merging the properties of multiple types, making them useful when you want to create more complex structures.
2. **Precise type definitions**: You can define types that represent multiple roles or entities at the same time, ensuring the object conforms to all required properties.

---

### **Comparison: Union vs Intersection Types**

| Feature               | **Union Type**                             | **Intersection Type**                          |
|-----------------------|--------------------------------------------|------------------------------------------------|
| **Definition**         | A variable can hold one of several types.  | A variable must hold all properties of combined types. |
| **Syntax**             | `A | B`                                   | `A & B`                                        |
| **Use case**           | Use when a variable can be one of multiple types. | Use when you want an object to have all properties of multiple types. |
| **Flexibility**        | More flexible, but less strict.            | More specific and strict.                     |
| **Type checking**      | TypeScript checks that the variable holds one of the union types. | TypeScript checks that the variable holds all the properties of all types. |

---

### **Use Cases for Union and Intersection Types**

#### **1. Union Types Use Case:**

Union types are useful when you want to allow a variable to accept multiple different types but don’t require it to adhere to all the types at once.

- **Example 1:**
  - A function that can accept either a string or a number for its input:

  ```typescript
  function formatValue(value: string | number): string {
    if (typeof value === "string") {
      return `String: ${value}`;
    } else {
      return `Number: ${value}`;
    }
  }

  console.log(formatValue(42)); // Output: "Number: 42"
  console.log(formatValue("Hello")); // Output: "String: Hello"
  ```

- **Example 2:**
  - Handling different shapes of data in a function (e.g., handling a `string` or an `array`):

  ```typescript
  function printLength(value: string | string[]): void {
    console.log(value.length);
  }

  printLength("Hello");  // Output: 5
  printLength(["A", "B", "C"]);  // Output: 3
  ```

#### **2. Intersection Types Use Case:**

Intersection types are useful when you need to combine multiple objects or interfaces to create a new object with all the properties of each.

- **Example 1:**
  - A function that accepts an object that needs to have properties from multiple interfaces:

  ```typescript
  interface Shape {
    color: string;
  }

  interface Circle {
    radius: number;
  }

  type ColoredCircle = Shape & Circle;

  function drawCircle(circle: ColoredCircle): void {
    console.log(`Drawing a ${circle.color} circle with radius ${circle.radius}`);
  }

  drawCircle({ color: "red", radius: 5 });
  ```

- **Example 2:**
  - Merging multiple configurations into a single configuration object:

  ```typescript
  interface UserConfig {
    username: string;
    email: string;
  }

  interface AdminConfig {
    adminLevel: number;
  }

  type AdminUser = UserConfig & AdminConfig;

  const admin: AdminUser = {
    username: "admin",
    email: "admin@example.com",
    adminLevel: 3,
  };
  ```

---

### **Advanced Use Cases**

#### **Using Union Types with Custom Types**:

You can combine union types with other TypeScript features like type aliases or interfaces for more precise control.

```typescript
type Status = "active" | "inactive" | "pending";

interface Task {
  id: number;
  status: Status;
}

let task: Task = {
  id: 1,
  status: "active",
};
```

#### **Using Intersection Types for Mixins**:

Intersection types are often used to implement "mixins" in TypeScript, allowing you to combine multiple classes or interfaces into one.

```typescript
class Vehicle {
  drive() {
    console.log("Driving a vehicle");
  }
}

class Airplane {
  fly() {
    console.log("Flying an airplane");
  }
}

type FlyingCar = Vehicle & Airplane;

const flyingCar: FlyingCar = {
  drive() {
    console.log("Driving a flying car");
  },
  fly() {
    console.log("Flying a flying car");
  },
};

flyingCar.drive();  // Output: Driving a flying car
flyingCar.fly();    // Output: Flying a flying car
```

---

### **Conclusion**

**Union types** and **intersection types** are powerful features in TypeScript that help you create more flexible and precise type definitions. Union types enable variables to accept multiple types, while intersection types combine multiple types into one. Both features allow you to design more robust and type-safe code by providing flexibility in how types can be used and combined.