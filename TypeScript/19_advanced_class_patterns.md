In TypeScript, advanced class patterns such as **Singleton**, **Factory**, and **Dependency Injection** are commonly used to manage object creation, maintain state, and inject dependencies into classes in a modular and scalable manner. Let’s explore each pattern in detail:

---

### **1. Singleton Pattern**

The **Singleton** pattern ensures that a class has only one instance, and provides a global point of access to that instance. It is useful for cases where you want to control access to shared resources, such as configuration settings or database connections.

#### **How It Works:**
- A **private static instance** is created within the class.
- A **public static method** is provided to retrieve the instance, creating it only when needed.
- The constructor is marked **private** to prevent instantiation from outside the class.

#### **Example:**
```typescript
class Singleton {
  // Static variable to hold the single instance
  private static instance: Singleton;

  // Private constructor to prevent direct instantiation
  private constructor() {}

  // Public static method to get the instance
  public static getInstance(): Singleton {
    if (!Singleton.instance) {
      Singleton.instance = new Singleton();
    }
    return Singleton.instance;
  }

  public sayHello(): void {
    console.log('Hello from Singleton!');
  }
}

// Usage
const singleton1 = Singleton.getInstance();
const singleton2 = Singleton.getInstance();

singleton1.sayHello();
console.log(singleton1 === singleton2);  // true
```

#### **Key Points:**
- **Private constructor** prevents creating multiple instances.
- **Static method** ensures that only one instance exists.
- The class is lazily instantiated (created only when needed).

---

### **2. Factory Pattern**

The **Factory** pattern is used to create objects without exposing the creation logic to the client. Instead of calling a constructor directly, a **factory method** is used to return an instance of the desired object.

#### **How It Works:**
- A **factory class** or method is responsible for creating objects.
- The client interacts with the factory, rather than creating objects directly, allowing for more flexibility in the object creation process.

#### **Example:**
```typescript
interface Product {
  name: string;
  price: number;
}

class ConcreteProductA implements Product {
  name = 'Product A';
  price = 100;
}

class ConcreteProductB implements Product {
  name = 'Product B';
  price = 150;
}

class ProductFactory {
  static createProduct(type: string): Product {
    switch (type) {
      case 'A':
        return new ConcreteProductA();
      case 'B':
        return new ConcreteProductB();
      default:
        throw new Error('Unknown product type');
    }
  }
}

// Usage
const productA = ProductFactory.createProduct('A');
console.log(productA.name, productA.price);  // Product A, 100

const productB = ProductFactory.createProduct('B');
console.log(productB.name, productB.price);  // Product B, 150
```

#### **Key Points:**
- The **Factory** encapsulates object creation, making it easier to manage and modify object instantiation.
- The **Factory method** can return different types of objects depending on the parameters, offering flexibility.

---

### **3. Dependency Injection (DI) Pattern**

The **Dependency Injection** pattern allows for the decoupling of class dependencies. Instead of a class creating its own dependencies, they are **injected** into the class from the outside, usually through its constructor. This increases testability and maintainability.

#### **How It Works:**
- **Constructor Injection**: Dependencies are passed through the constructor.
- **Property Injection**: Dependencies are assigned to properties.
- **Method Injection**: Dependencies are passed as method arguments.

#### **Example:**
```typescript
class DatabaseService {
  connect(): void {
    console.log('Connected to the database');
  }
}

class UserService {
  // Injecting the dependency through constructor
  constructor(private dbService: DatabaseService) {}

  getUserData(): void {
    this.dbService.connect();
    console.log('Fetching user data');
  }
}

// Usage
const dbService = new DatabaseService();
const userService = new UserService(dbService);

userService.getUserData();  
// Output: 
// Connected to the database
// Fetching user data
```

#### **Key Points:**
- **Loose Coupling**: Classes don't need to know how their dependencies are created.
- **Testability**: You can easily replace dependencies with mocks or stubs for testing.
- **Flexibility**: Dependencies can be configured externally (e.g., through a DI container or configuration).

---

### **4. Dependency Injection with a DI Container (Advanced)**

For more complex applications, you can use a **DI container** to manage object creation and dependency resolution automatically.

#### **Example with a Simple DI Container:**
```typescript
class DIContainer {
  private static container: Map<any, any> = new Map();

  static register<T>(token: any, service: T): void {
    DIContainer.container.set(token, service);
  }

  static resolve<T>(token: any): T {
    const service = DIContainer.container.get(token);
    if (!service) {
      throw new Error(`Service ${token} not found`);
    }
    return service;
  }
}

class Logger {
  log(message: string): void {
    console.log('Log:', message);
  }
}

class UserService {
  constructor(private logger: Logger) {}

  getUserData(): void {
    this.logger.log('Fetching user data');
  }
}

// Register services in the DI container
DIContainer.register(Logger, new Logger());
DIContainer.register(UserService, new UserService(DIContainer.resolve(Logger)));

// Resolving and using the service
const userService = DIContainer.resolve(UserService);
userService.getUserData();  // Output: Log: Fetching user data
```

#### **Key Points:**
- The **DI container** is used to manage services and their dependencies, making it easier to scale.
- **Automatic dependency resolution** ensures that each service gets the required dependencies.
- This approach is common in larger applications or frameworks like **Angular**.

---

### **Comparison and When to Use These Patterns**

- **Singleton**: Use when you need a single instance of a class throughout your application, such as for configuration or shared resources.
- **Factory**: Use when the process of creating objects is complex or varies, allowing the client to be decoupled from the instantiation logic.
- **Dependency Injection**: Use when you want to decouple components, improve testability, and manage dependencies more easily. DI is especially useful in large, complex systems where components may depend on external services or resources.

By leveraging these advanced class patterns, you can enhance code organization, flexibility, and maintainability in TypeScript applications.