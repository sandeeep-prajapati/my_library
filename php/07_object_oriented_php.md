# **Object-Oriented Programming (OOP) in PHP**  

Object-Oriented Programming (OOP) is a programming paradigm based on objects and classes. PHP supports OOP, allowing for modular, reusable, and scalable code.  

---

## **1. Basics of OOP in PHP**  

### **a) Classes and Objects**  

- A **class** is a blueprint for creating objects.  
- An **object** is an instance of a class.  

**Example:**  

```php
<?php
class Car {
    public $brand;
    
    public function setBrand($name) {
        $this->brand = $name;
    }

    public function getBrand() {
        return $this->brand;
    }
}

// Creating an object
$myCar = new Car();
$myCar->setBrand("Tesla");
echo $myCar->getBrand();  // Output: Tesla
?>
```

### **b) Properties and Methods**  

- **Properties** (variables inside a class) define an object’s attributes.  
- **Methods** (functions inside a class) define an object’s behavior.  

```php
<?php
class Person {
    public $name;  // Property

    public function sayHello() {  // Method
        return "Hello, my name is " . $this->name;
    }
}

$p = new Person();
$p->name = "Sandeep";
echo $p->sayHello();  // Output: Hello, my name is Sandeep
?>
```

---

## **2. OOP Concepts in PHP**  

### **a) Constructor and Destructor**  

- **`__construct()`** is called automatically when an object is created.  
- **`__destruct()`** is called when an object is destroyed.  

```php
<?php
class Animal {
    public $type;

    public function __construct($type) {
        $this->type = $type;
        echo "A new $type has been created!<br>";
    }

    public function __destruct() {
        echo "The $this->type is being removed.<br>";
    }
}

$cat = new Animal("Cat");  // Output: A new Cat has been created!
?>
```

---

### **b) Inheritance**  

Inheritance allows a class (child) to derive properties and methods from another class (parent).  

```php
<?php
class Vehicle {
    public $color;

    public function setColor($color) {
        $this->color = $color;
    }

    public function getColor() {
        return $this->color;
    }
}

// Child class inherits from Vehicle
class Car extends Vehicle {
    public $brand;

    public function setBrand($brand) {
        $this->brand = $brand;
    }

    public function getCarInfo() {
        return "Brand: $this->brand, Color: $this->color";
    }
}

$myCar = new Car();
$myCar->setBrand("Toyota");
$myCar->setColor("Red");
echo $myCar->getCarInfo();  // Output: Brand: Toyota, Color: Red
?>
```

---

### **c) Polymorphism**  

Polymorphism allows methods in child classes to override methods in parent classes.  

```php
<?php
class Animal {
    public function makeSound() {
        return "Some sound...";
    }
}

class Dog extends Animal {
    public function makeSound() {
        return "Bark!";
    }
}

$dog = new Dog();
echo $dog->makeSound();  // Output: Bark!
?>
```

---

### **d) Encapsulation**  

Encapsulation restricts access to class properties using **access modifiers**:  
- **`public`** – Accessible from anywhere.  
- **`private`** – Accessible only inside the class.  
- **`protected`** – Accessible within the class and its child classes.  

```php
<?php
class BankAccount {
    private $balance = 0;

    public function deposit($amount) {
        $this->balance += $amount;
    }

    public function getBalance() {
        return $this->balance;
    }
}

$account = new BankAccount();
$account->deposit(500);
echo $account->getBalance();  // Output: 500
?>
```

---

### **e) Interfaces**  

An interface defines methods that must be implemented in a class.  

```php
<?php
interface Animal {
    public function makeSound();
}

class Cat implements Animal {
    public function makeSound() {
        return "Meow!";
    }
}

$cat = new Cat();
echo $cat->makeSound();  // Output: Meow!
?>
```

---

## **3. Summary**  

| Concept       | Description |
|--------------|-------------|
| **Class**     | Blueprint for objects |
| **Object**    | Instance of a class |
| **Constructor** | Automatically runs when an object is created |
| **Inheritance** | Child class inherits from a parent class |
| **Polymorphism** | Child class overrides a parent method |
| **Encapsulation** | Restricts access to class properties |
| **Interfaces** | Define required methods for a class |

Would you like some **practice exercises**? 🚀