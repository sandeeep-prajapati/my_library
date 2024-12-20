### **Decorators in TypeScript**

Decorators in TypeScript are a special kind of declaration that can be attached to classes, methods, properties, or parameters. They allow you to modify or enhance the behavior of the target entity (class, method, property, or parameter) in a declarative way. Decorators are a form of **meta-programming** and are often used for various purposes such as logging, validation, access control, or binding methods.

Decorators are **experimental** in TypeScript, meaning they require enabling the `experimentalDecorators` option in the `tsconfig.json` file.

### **How to Enable Decorators:**

To use decorators in TypeScript, you need to enable the `experimentalDecorators` compiler option:

```json
{
  "compilerOptions": {
    "experimentalDecorators": true
  }
}
```

### **Types of Decorators**

There are four main types of decorators in TypeScript:

1. **Class Decorators** – Applied to a class.
2. **Method Decorators** – Applied to methods.
3. **Property Decorators** – Applied to properties.
4. **Parameter Decorators** – Applied to function parameters.

### **Class Decorators**

A **class decorator** is a function that is applied to the class constructor. It can be used to modify the class or its prototype.

#### Example:

```typescript
function logClass(target: Function) {
  console.log(`Class created: ${target.name}`);
}

@logClass
class MyClass {
  constructor() {
    console.log("MyClass instance created");
  }
}

const instance = new MyClass();
// Output:
// Class created: MyClass
// MyClass instance created
```

In this example, the `logClass` decorator is applied to the `MyClass` class, and it logs the class name whenever an instance of the class is created.

### **Method Decorators**

A **method decorator** is a function applied to methods within a class. It can modify the method’s behavior, such as logging method calls or modifying return values.

#### Example:

```typescript
function logMethod(target: any, propertyName: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  
  descriptor.value = function (...args: any[]) {
    console.log(`Calling method: ${propertyName} with arguments: ${args}`);
    return originalMethod.apply(this, args);
  };

  return descriptor;
}

class MyClass {
  @logMethod
  sayHello(name: string) {
    console.log(`Hello, ${name}!`);
  }
}

const instance = new MyClass();
instance.sayHello("Alice");
// Output:
// Calling method: sayHello with arguments: ["Alice"]
// Hello, Alice!
```

Here, the `logMethod` decorator intercepts calls to the `sayHello` method and logs the arguments before calling the original method.

### **Property Decorators**

A **property decorator** is applied to a property of a class. It can be used to modify the behavior of the property, such as adding validation, changing property descriptors, or defining getter/setter functions.

#### Example:

```typescript
function readonly(target: any, propertyKey: string) {
  const descriptor = Object.getOwnPropertyDescriptor(target, propertyKey);
  if (descriptor) {
    descriptor.writable = false;  // Make the property read-only
    Object.defineProperty(target, propertyKey, descriptor);
  }
}

class MyClass {
  @readonly
  name: string;

  constructor(name: string) {
    this.name = name;
  }
}

const instance = new MyClass("Alice");
console.log(instance.name); // Alice
instance.name = "Bob"; // Error: Cannot assign to 'name' because it is a read-only property.
```

In this example, the `readonly` decorator prevents the `name` property from being modified after it is initialized.

### **Parameter Decorators**

A **parameter decorator** is used to modify or enhance the behavior of function parameters. It is applied to parameters of class methods and can be used for purposes such as parameter validation or logging.

#### Example:

```typescript
function logParameter(target: any, methodName: string, parameterIndex: number) {
  const existingParameters = Reflect.getOwnMetadata('parameters', target, methodName) || [];
  existingParameters.push(parameterIndex);
  Reflect.defineMetadata('parameters', existingParameters, target, methodName);
}

class MyClass {
  greet(@logParameter name: string) {
    console.log(`Hello, ${name}!`);
  }
}

const instance = new MyClass();
instance.greet("Alice");
```

In this example, the `logParameter` decorator stores the index of the parameter. You could extend this to perform actions like logging parameter values.

### **Using Multiple Decorators**

You can apply multiple decorators to a single class or method. The order of the decorators is important because they are applied in **bottom-up** order (i.e., from the last decorator to the first).

#### Example:

```typescript
function firstDecorator(target: any) {
  console.log("First decorator applied");
}

function secondDecorator(target: any) {
  console.log("Second decorator applied");
}

@firstDecorator
@secondDecorator
class MyClass {}

const instance = new MyClass();
// Output:
// Second decorator applied
// First decorator applied
```

In this case, the `secondDecorator` is applied before the `firstDecorator`.

### **Decorator Factories**

A **decorator factory** is a function that returns a decorator. This is useful when you want to pass parameters to the decorator.

#### Example:

```typescript
function logMessage(message: string) {
  return function (target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    descriptor.value = function (...args: any[]) {
      console.log(`${message}: ${args}`);
      return originalMethod.apply(this, args);
    };
  };
}

class MyClass {
  @logMessage("Method called")
  sayHello(name: string) {
    console.log(`Hello, ${name}`);
  }
}

const instance = new MyClass();
instance.sayHello("Alice");
// Output:
// Method called: ["Alice"]
// Hello, Alice
```

In this example, `logMessage` is a decorator factory that allows you to pass a message as an argument to the decorator.

### **Conclusion**

Decorators in TypeScript are a powerful feature that allows you to add metadata, modify class behavior, and enhance methods, properties, or parameters. They enable you to write cleaner, more reusable code for concerns such as logging, validation, access control, and more. While decorators are still an experimental feature in TypeScript, they provide a compelling way to add functionality to classes and methods in a declarative and readable way.