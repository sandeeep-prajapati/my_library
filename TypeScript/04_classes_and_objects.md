TypeScript extends JavaScript by adding static typing and other features to support object-oriented programming (OOP) concepts, such as **classes**, **inheritance**, and **polymorphism**. Here’s how TypeScript handles these core OOP concepts:

### 1. **Classes in TypeScript**
In TypeScript, **classes** are used to define object structures and behavior. TypeScript’s class system is similar to JavaScript ES6 classes but with additional features like type annotations for properties and method parameters.

- **Basic class definition**:
  ```typescript
  class Animal {
    name: string;

    constructor(name: string) {
      this.name = name;
    }

    makeSound(): void {
      console.log(`${this.name} makes a sound`);
    }
  }

  const dog = new Animal("Dog");
  dog.makeSound();  // Output: Dog makes a sound
  ```

- **Access modifiers**: TypeScript introduces access modifiers (`public`, `private`, and `protected`) to control the visibility of properties and methods.

  - `public` (default): The property or method is accessible from anywhere.
  - `private`: The property or method is accessible only within the class.
  - `protected`: The property or method is accessible within the class and subclasses.

  ```typescript
  class Animal {
    public name: string;  // Public property
    private age: number;  // Private property

    constructor(name: string, age: number) {
      this.name = name;
      this.age = age;
    }

    public greet(): void {
      console.log(`Hello, I am ${this.name}`);
    }
  }

  const dog = new Animal("Dog", 5);
  dog.greet();  // Output: Hello, I am Dog
  // dog.age;     // Error: Property 'age' is private and only accessible within class 'Animal'.
  ```

### 2. **Inheritance in TypeScript**
TypeScript supports **inheritance** using the `extends` keyword, allowing a class to inherit from another class. This enables **code reuse** and the ability to build more specific classes from more general ones.

- **Class inheritance**:
  ```typescript
  class Animal {
    name: string;

    constructor(name: string) {
      this.name = name;
    }

    makeSound(): void {
      console.log(`${this.name} makes a sound`);
    }
  }

  class Dog extends Animal {
    breed: string;

    constructor(name: string, breed: string) {
      super(name);  // Call the parent class constructor
      this.breed = breed;
    }

    makeSound(): void {
      console.log(`${this.name} barks`);
    }
  }

  const dog = new Dog("Rex", "German Shepherd");
  dog.makeSound();  // Output: Rex barks
  ```

In this example:
- The `Dog` class extends `Animal`, inheriting its properties and methods.
- The `super()` function is used to call the constructor of the parent class (`Animal`), which allows the `Dog` class to inherit the `name` property.
- The `makeSound()` method is **overridden** in the `Dog` class to provide specific behavior for the `Dog` class.

### 3. **Polymorphism in TypeScript**
**Polymorphism** refers to the ability of different classes to implement methods with the same name but potentially different behavior. In TypeScript, this is often achieved through **method overriding** and **method overloading**.

- **Method Overriding**:
  In TypeScript, method overriding is achieved when a subclass redefines a method that was inherited from its parent class.

  ```typescript
  class Animal {
    makeSound(): void {
      console.log("Animal makes a sound");
    }
  }

  class Dog extends Animal {
    makeSound(): void {
      console.log("Dog barks");
    }
  }

  class Cat extends Animal {
    makeSound(): void {
      console.log("Cat meows");
    }
  }

  const dog = new Dog();
  dog.makeSound();  // Output: Dog barks

  const cat = new Cat();
  cat.makeSound();  // Output: Cat meows
  ```

In this example:
- Both `Dog` and `Cat` override the `makeSound()` method inherited from `Animal`. This is an example of **polymorphism** where the method name is the same, but the behavior is different for different classes.

- **Method Overloading**: TypeScript also allows you to define multiple signatures for a function, which is another form of polymorphism. However, unlike other OOP languages, TypeScript does not support multiple method definitions in a single class body directly, but you can simulate method overloading using type annotations.

  ```typescript
  class Printer {
    print(value: string): void;
    print(value: number): void;
    print(value: any): void {
      console.log(value);
    }
  }

  const printer = new Printer();
  printer.print("Hello");  // Output: Hello
  printer.print(42);       // Output: 42
  ```

In this example, the `print()` method has different signatures for `string` and `number`, allowing the method to handle different types of inputs.

### 4. **Abstract Classes in TypeScript**
TypeScript allows you to define **abstract classes**, which cannot be instantiated directly but can serve as a blueprint for subclasses. Abstract classes can define methods that must be implemented by subclasses.

- **Abstract class and methods**:
  ```typescript
  abstract class Animal {
    abstract makeSound(): void;  // Abstract method

    move(): void {
      console.log("Moving...");
    }
  }

  class Dog extends Animal {
    makeSound(): void {
      console.log("Barking");
    }
  }

  const dog = new Dog();
  dog.makeSound();  // Output: Barking
  ```

In this example:
- `Animal` is an **abstract class** that defines an abstract method `makeSound()` that must be implemented by any subclass (like `Dog`).
- The `move()` method is a concrete method, meaning it can be used directly by subclasses.

### 5. **Interfaces in TypeScript for Polymorphism**
TypeScript interfaces can be used for **polymorphism** as well. An interface defines the contract for classes that implement it, ensuring that different classes follow the same structure while allowing for different implementations.

- **Interface and polymorphism**:
  ```typescript
  interface SoundMaker {
    makeSound(): void;
  }

  class Dog implements SoundMaker {
    makeSound(): void {
      console.log("Barking");
    }
  }

  class Cat implements SoundMaker {
    makeSound(): void {
      console.log("Meowing");
    }
  }

  function playSound(animal: SoundMaker): void {
    animal.makeSound();
  }

  const dog = new Dog();
  const cat = new Cat();

  playSound(dog);  // Output: Barking
  playSound(cat);  // Output: Meowing
  ```

In this example, both `Dog` and `Cat` implement the `SoundMaker` interface. The `playSound()` function takes any object that implements the `SoundMaker` interface, demonstrating polymorphism with different object types.

---

### **Conclusion:**
TypeScript provides robust support for object-oriented programming through **classes**, **inheritance**, and **polymorphism**. It enhances JavaScript’s OOP capabilities by adding features like **type annotations**, **access modifiers**, and **abstract classes**, making it easier to build and maintain object-oriented systems. By leveraging these features, you can write more structured, scalable, and maintainable code in TypeScript.