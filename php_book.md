## **Basic Syntax of PHP**  

PHP scripts are executed on the server, and the output is sent to the browser. The syntax is simple, resembling C, Java, and Perl, making it easy to learn.

---

### **1. PHP Script Structure**
A PHP script starts with `<?php` and ends with `?>`. The script can be embedded inside an HTML file.  

**Example:**  
```php
<!DOCTYPE html>
<html>
<head>
    <title>My First PHP Script</title>
</head>
<body>
    <h1><?php echo "Hello, World!"; ?></h1>
</body>
</html>
```
- The `echo` statement outputs text to the webpage.  

---

### **2. PHP Variables**  
- Variables in PHP start with `$`.  
- They are case-sensitive.  
- No need to declare the type explicitly (PHP is loosely typed).  

**Example:**  
```php
<?php
$name = "Sandeep";
$age = 25;
echo "My name is $name and I am $age years old.";
?>
```

---

### **3. Data Types in PHP**  
PHP supports several data types:  

| Data Type  | Example |
|------------|---------|
| **String** | `$name = "Sandeep";` |
| **Integer** | `$age = 25;` |
| **Float (Double)** | `$price = 99.99;` |
| **Boolean** | `$isActive = true;` |
| **Array** | `$colors = array("Red", "Green", "Blue");` |
| **Object** | `$car = new Car();` |
| **NULL** | `$value = NULL;` |

**Example:**  
```php
<?php
$price = 499.99;
$isAvailable = true;
$colors = ["Red", "Blue", "Green"];
echo "Price: $price <br>";
echo "Available: " . ($isAvailable ? "Yes" : "No") . "<br>";
echo "First color: " . $colors[0];
?>
```

---

### **4. Constants in PHP**  
- Constants are defined using `define()` or `const`.  
- Once defined, they cannot be changed.  

**Example:**  
```php
<?php
define("SITE_NAME", "MyWebsite");
const PI = 3.1416;
echo "Welcome to " . SITE_NAME . "<br>";
echo "Value of PI: " . PI;
?>
```

---

### **5. Comments in PHP**  
- **Single-line comment:** `//` or `#`  
- **Multi-line comment:** `/* ... */`  

**Example:**  
```php
<?php
// This is a single-line comment
# Another single-line comment

/*
This is a multi-line comment.
It spans multiple lines.
*/

echo "PHP comments example.";
?>
```

## **Control Structures in PHP**  

Control structures in PHP allow the execution of different code blocks based on conditions and loops. They include **conditionals** (`if`, `else`, `switch`) and **loops** (`for`, `while`, `foreach`).  

---

## **1. Conditional Statements**  

### **a) `if` Statement**  
Executes a block of code only if a condition is `true`.  

**Example:**  
```php
<?php
$age = 20;
if ($age >= 18) {
    echo "You are eligible to vote.";
}
?>
```

### **b) `if-else` Statement**  
Executes one block if the condition is `true`, otherwise executes another block.  

**Example:**  
```php
<?php
$marks = 40;
if ($marks >= 50) {
    echo "You passed!";
} else {
    echo "You failed.";
}
?>
```

### **c) `if-elseif-else` Statement**  
Allows multiple conditions to be checked.  

**Example:**  
```php
<?php
$score = 85;
if ($score >= 90) {
    echo "Grade: A+";
} elseif ($score >= 80) {
    echo "Grade: A";
} elseif ($score >= 70) {
    echo "Grade: B";
} else {
    echo "Grade: C";
}
?>
```

### **d) `switch` Statement**  
Used when there are multiple possible conditions for a variable.  

**Example:**  
```php
<?php
$day = "Tuesday";

switch ($day) {
    case "Monday":
        echo "Start of the workweek.";
        break;
    case "Friday":
        echo "Weekend is near!";
        break;
    case "Sunday":
        echo "It's a rest day!";
        break;
    default:
        echo "It's a normal day.";
}
?>
```

---

## **2. Loops in PHP**  

Loops allow executing a block of code multiple times.

### **a) `for` Loop**  
Used when the number of iterations is known.  

**Syntax:**  
```php
for (initialization; condition; increment/decrement) {
    // Code to execute
}
```

**Example:**  
```php
<?php
for ($i = 1; $i <= 5; $i++) {
    echo "Number: $i <br>";
}
?>
```

### **b) `while` Loop**  
Used when the number of iterations is unknown but depends on a condition.  

**Syntax:**  
```php
while (condition) {
    // Code to execute
}
```

**Example:**  
```php
<?php
$x = 1;
while ($x <= 5) {
    echo "Value: $x <br>";
    $x++;
}
?>
```

### **c) `do-while` Loop**  
Executes the block at least once, then checks the condition.  

**Example:**  
```php
<?php
$y = 1;
do {
    echo "Value: $y <br>";
    $y++;
} while ($y <= 5);
?>
```

### **d) `foreach` Loop**  
Used to iterate through arrays.  

**Example:**  
```php
<?php
$colors = ["Red", "Blue", "Green"];
foreach ($colors as $color) {
    echo "Color: $color <br>";
}
?>
```
## **Functions in PHP**  

Functions are reusable blocks of code in PHP that perform specific tasks. They improve code organization and reduce repetition.  

---

## **1. Function Declaration in PHP**  
A function is defined using the `function` keyword, followed by a name and optional parameters.  

### **Basic Function Declaration**  
```php
<?php
function greet() {
    echo "Hello, World!";
}
greet(); // Calling the function
?>
```
**Output:**  
```
Hello, World!
```

---

## **2. Function Parameters**  
Functions can accept parameters (inputs) to process dynamic values.  

### **Function with Parameters**  
```php
<?php
function greetUser($name) {
    echo "Hello, $name!";
}
greetUser("Sandeep");
?>
```
**Output:**  
```
Hello, Sandeep!
```

### **Default Parameter Value**  
If a parameter is not provided, a default value is used.  
```php
<?php
function sayHello($name = "Guest") {
    echo "Hello, $name!";
}
sayHello(); // Uses default value
sayHello("Sandeep");
?>
```
**Output:**  
```
Hello, Guest!
Hello, Sandeep!
```

### **Multiple Parameters**  
```php
<?php
function add($a, $b) {
    echo $a + $b;
}
add(5, 10);
?>
```
**Output:**  
```
15
```

---

## **3. Return Values**  
Functions can return values instead of printing them.  

### **Returning a Value**  
```php
<?php
function multiply($a, $b) {
    return $a * $b;
}
$result = multiply(4, 5);
echo "Result: $result";
?>
```
**Output:**  
```
Result: 20
```

---

## **4. Variable Scope in PHP**  
Scope determines where variables can be accessed.  

### **a) Local Scope**  
Variables declared inside a function are not accessible outside.  
```php
<?php
function testScope() {
    $x = 10; // Local variable
    echo "Inside function: $x";
}
testScope();
// echo $x; // This will cause an error (undefined variable)
?>
```

### **b) Global Scope**  
Variables declared outside functions are **not** accessible inside unless `global` is used.  
```php
<?php
$y = 20; // Global variable

function testGlobal() {
    global $y; // Accessing global variable
    echo "Inside function: $y";
}
testGlobal();
?>
```

### **c) Static Variables**  
A `static` variable retains its value across function calls.  
```php
<?php
function counter() {
    static $count = 0;
    $count++;
    echo "Count: $count <br>";
}
counter();
counter();
counter();
?>
```
**Output:**  
```
Count: 1  
Count: 2  
Count: 3  
```
If we remove `static`, `$count` would reset to `0` on each call.
# **PHP Arrays and Their Types**  

Arrays in PHP store multiple values in a single variable. They are **flexible**, allowing various data types and operations.  

---

## **1. Types of Arrays in PHP**  

### **a) Indexed Arrays (Numeric Arrays)**  
These arrays use numeric indexes, starting from `0`.  

**Declaration:**  
```php
<?php
$fruits = ["Apple", "Banana", "Orange"];
echo $fruits[0]; // Output: Apple
?>
```

**Looping through an Indexed Array:**  
```php
<?php
$fruits = ["Mango", "Papaya", "Grapes"];
foreach ($fruits as $fruit) {
    echo "$fruit <br>";
}
?>
```

---

### **b) Associative Arrays**  
These arrays use named keys instead of numeric indexes.  

**Declaration:**  
```php
<?php
$student = [
    "name" => "Sandeep",
    "age" => 22,
    "course" => "PHP"
];
echo $student["name"]; // Output: Sandeep
?>
```

**Looping through an Associative Array:**  
```php
<?php
$student = ["name" => "John", "age" => 25, "city" => "New York"];
foreach ($student as $key => $value) {
    echo "$key: $value <br>";
}
?>
```

---

### **c) Multidimensional Arrays**  
These arrays contain nested arrays.  

**Declaration:**  
```php
<?php
$employees = [
    ["John", 25, "Developer"],
    ["Sara", 30, "Designer"],
    ["Mike", 28, "Manager"]
];
echo $employees[0][0]; // Output: John
?>
```

**Looping through a Multidimensional Array:**  
```php
<?php
$employees = [
    ["John", 25, "Developer"],
    ["Sara", 30, "Designer"],
    ["Mike", 28, "Manager"]
];

foreach ($employees as $employee) {
    echo "Name: $employee[0], Age: $employee[1], Role: $employee[2] <br>";
}
?>
```

---

## **2. Manipulating Arrays with Built-in Functions**  

### **a) Adding Elements to an Array**  
- `array_push($array, value)`: Adds an element to the end.  
- `$array[] = value`: Alternative method.  
- `array_unshift($array, value)`: Adds an element to the beginning.  

```php
<?php
$colors = ["Red", "Blue"];
array_push($colors, "Green"); // Adds at the end
array_unshift($colors, "Yellow"); // Adds at the beginning
print_r($colors);
?>
```
**Output:**  
```
Array ( [0] => Yellow [1] => Red [2] => Blue [3] => Green )
```

---

### **b) Removing Elements from an Array**  
- `array_pop($array)`: Removes the last element.  
- `array_shift($array)`: Removes the first element.  

```php
<?php
$animals = ["Lion", "Tiger", "Elephant"];
array_pop($animals); // Removes "Elephant"
array_shift($animals); // Removes "Lion"
print_r($animals);
?>
```
**Output:**  
```
Array ( [0] => Tiger )
```

---

### **c) Sorting Arrays**  
- `sort($array)`: Sorts an indexed array in ascending order.  
- `rsort($array)`: Sorts in descending order.  
- `asort($array)`: Sorts an associative array by values.  
- `ksort($array)`: Sorts an associative array by keys.  

```php
<?php
$numbers = [4, 2, 8, 1];
sort($numbers);
print_r($numbers);
?>
```
**Output:**  
```
Array ( [0] => 1 [1] => 2 [2] => 4 [3] => 8 )
```

---

### **d) Checking if a Value Exists**  
- `in_array(value, $array)`: Checks if a value is in the array.  

```php
<?php
$fruits = ["Apple", "Banana", "Orange"];
if (in_array("Banana", $fruits)) {
    echo "Banana is in the array!";
}
?>
```
**Output:**  
```
Banana is in the array!
```

---

### **e) Counting Elements in an Array**  
- `count($array)`: Returns the number of elements.  

```php
<?php
$items = ["Pen", "Notebook", "Eraser"];
echo count($items); // Output: 3
?>
```
# **PHP Strings and String Manipulation Functions**  

A **string** in PHP is a sequence of characters. Strings are widely used in PHP for handling text, processing user inputs, and dynamic content generation.  

---

## **1. Creating Strings in PHP**  

PHP supports single quotes (`'`) and double quotes (`"`) for defining strings:  

```php
<?php
$name = "Sandeep";
echo 'Hello, $name';  // Output: Hello, $name (single quotes do not parse variables)
echo "Hello, $name";  // Output: Hello, Sandeep (double quotes parse variables)
?>
```

### **String Concatenation**  
```php
<?php
$first = "Hello";
$second = "World";
echo $first . " " . $second;  // Output: Hello World
?>
```
---

## **2. Important String Functions in PHP**  

### **a) `strlen()` – Get the Length of a String**  
The `strlen()` function returns the number of characters (including spaces).  

```php
<?php
$text = "Hello, PHP!";
echo strlen($text);  // Output: 11
?>
```

---

### **b) `str_replace()` – Replace a Word in a String**  
The `str_replace()` function replaces occurrences of a substring with another.  

```php
<?php
$text = "I love JavaScript!";
$updatedText = str_replace("JavaScript", "PHP", $text);
echo $updatedText;  // Output: I love PHP!
?>
```

---

### **c) `substr()` – Extract Part of a String**  
The `substr()` function extracts a portion of a string.  

```php
<?php
$text = "Programming";
echo substr($text, 0, 5);   // Output: Progr (First 5 characters)
echo substr($text, -3);     // Output: ing (Last 3 characters)
?>
```

---

### **d) `explode()` – Convert a String into an Array**  
The `explode()` function splits a string into an array using a delimiter.  

```php
<?php
$sentence = "PHP is fun to learn";
$words = explode(" ", $sentence);
print_r($words);
?>
```
**Output:**  
```
Array ( [0] => PHP [1] => is [2] => fun [3] => to [4] => learn )
```

---

## **3. Other Useful String Functions**  

### **e) `implode()` – Convert an Array to a String**  
```php
<?php
$words = ["PHP", "is", "awesome"];
$sentence = implode(" ", $words);
echo $sentence;  // Output: PHP is awesome
?>
```

### **f) `strtolower()` and `strtoupper()` – Convert Case**  
```php
<?php
$text = "Hello World";
echo strtolower($text);  // Output: hello world
echo strtoupper($text);  // Output: HELLO WORLD
?>
```

### **g) `trim()` – Remove Spaces from a String**  
```php
<?php
$text = "   PHP is great!   ";
echo trim($text);  // Output: PHP is great!
?>
```

### **h) `strpos()` – Find a Substring Position**  
```php
<?php
$text = "Hello PHP";
echo strpos($text, "PHP");  // Output: 6
?>
```

---
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
# **Error Handling in PHP**  

Error handling in PHP allows developers to gracefully manage and debug errors instead of letting them crash the application. PHP provides several mechanisms for handling errors, including `try-catch` blocks, custom error handlers, and error reporting functions.  

---

## **1. Types of Errors in PHP**  
PHP has different types of errors:  

| **Error Type** | **Description** |
|--------------|-------------|
| **Fatal Error** | Stops script execution (e.g., calling a non-existent function). |
| **Warning** | Doesn't stop execution but signals an issue (e.g., `include()` missing file). |
| **Notice** | Indicates minor errors (e.g., using an undefined variable). |
| **Parse Error** | Occurs due to syntax errors. |
| **Deprecated Error** | Warns about usage of outdated functions. |

---

## **2. Using `try`, `catch`, and `finally` for Exception Handling**  

PHP allows handling exceptions using `try`, `catch`, and `finally` blocks.  

### **a) Basic `try-catch` Block**  

```php
<?php
function divide($num1, $num2) {
    if ($num2 == 0) {
        throw new Exception("Division by zero is not allowed.");
    }
    return $num1 / $num2;
}

try {
    echo divide(10, 0);
} catch (Exception $e) {
    echo "Error: " . $e->getMessage();
}
?>
```
**Output:**  
`Error: Division by zero is not allowed.`  

### **b) Using `finally`**  

The `finally` block always executes, regardless of whether an exception occurs.  

```php
<?php
try {
    echo divide(10, 2);
} catch (Exception $e) {
    echo "Error: " . $e->getMessage();
} finally {
    echo "<br>Execution completed.";
}
?>
```
**Output:**  
`5`  
`Execution completed.`  

---

## **3. Custom Exception Handling**  

You can create your own exception classes by extending the `Exception` class.  

```php
<?php
class MyException extends Exception {}

try {
    throw new MyException("This is a custom exception.");
} catch (MyException $e) {
    echo "Caught: " . $e->getMessage();
}
?>
```

---

## **4. Error Reporting Functions**  

### **a) `error_reporting()`**  
Controls which errors PHP should report.  

```php
<?php
error_reporting(E_ALL); // Report all errors
?>
```

### **b) `set_error_handler()`**  
Define a custom function to handle errors.  

```php
<?php
function customErrorHandler($errno, $errstr, $errfile, $errline) {
    echo "Error [$errno]: $errstr in $errfile on line $errline";
}

// Set the custom error handler
set_error_handler("customErrorHandler");

// Trigger an error
echo $undefinedVar;  // Notice: Undefined variable
?>
```

---

## **5. Logging Errors with `error_log()`**  

You can log errors to a file instead of displaying them.  

```php
<?php
error_log("This is an error message!", 3, "errors.log");
?>
```
This logs the error in the `errors.log` file.

---

## **6. Disabling Error Display in Production**  

In production environments, disable error display to users.  

```php
<?php
ini_set('display_errors', 0);
ini_set('log_errors', 1);
ini_set('error_log', 'errors.log'); // Log errors to a file
?>
```

---
# **PHP File Handling**  

PHP provides several built-in functions to interact with files, allowing developers to create, read, write, and delete files on a server. Proper file handling is crucial for storing and retrieving data dynamically.  

---

## **1. Opening a File with `fopen()`**  

The `fopen()` function is used to open a file in different modes:  

```php
<?php
$file = fopen("example.txt", "r"); // Opens a file in read mode
?>
```

### **Common File Modes in `fopen()`**  

| **Mode** | **Description** |
|----------|-------------|
| `"r"`  | Read-only; file must exist. |
| `"w"`  | Write-only; erases content if file exists, creates a new file if it doesn’t. |
| `"a"`  | Append mode; writes at the end of the file. |
| `"x"`  | Creates a new file for writing; fails if file exists. |
| `"r+"` | Read & write; file must exist. |
| `"w+"` | Read & write; erases file contents. |
| `"a+"` | Read & write; appends data. |

---

## **2. Writing to a File with `fwrite()`**  

The `fwrite()` function writes data into an open file.  

```php
<?php
$file = fopen("example.txt", "w");  // Open file in write mode
fwrite($file, "Hello, PHP File Handling!"); // Write data
fclose($file); // Close file
?>
```

If the file **does not exist**, PHP creates it automatically.

---

## **3. Reading from a File with `fread()`**  

The `fread()` function reads a specific number of bytes from a file.  

```php
<?php
$file = fopen("example.txt", "r");  // Open file in read mode
$content = fread($file, filesize("example.txt")); // Read entire file
fclose($file); // Close file
echo $content; // Display content
?>
```

---

## **4. Reading a File with `file_get_contents()`**  

The `file_get_contents()` function reads the entire file into a string. It's simpler than `fread()`.  

```php
<?php
$content = file_get_contents("example.txt");
echo $content;
?>
```

✅ **Best for reading entire files in one step.**  

---

## **5. Appending Data to a File (`a` mode)**  

You can **append** data without erasing existing content using `a` mode in `fopen()`.  

```php
<?php
$file = fopen("example.txt", "a");
fwrite($file, "\nAppending new data!");
fclose($file);
?>
```

---

## **6. Checking if a File Exists**  

Before opening a file, it's good practice to check if it exists using `file_exists()`.  

```php
<?php
if (file_exists("example.txt")) {
    echo "File exists!";
} else {
    echo "File not found!";
}
?>
```

---

## **7. Deleting a File with `unlink()`**  

The `unlink()` function deletes a file.  

```php
<?php
if (file_exists("example.txt")) {
    unlink("example.txt");
    echo "File deleted!";
} else {
    echo "File does not exist!";
}
?>
```

---

## **8. Reading a File Line-by-Line with `fgets()`**  

To read a file **line-by-line**, use `fgets()`.  

```php
<?php
$file = fopen("example.txt", "r");
while (!feof($file)) {  // Loop until end of file
    echo fgets($file) . "<br>";
}
fclose($file);
?>
```

---

## **9. Locking a File with `flock()`**  

To prevent multiple processes from modifying the same file simultaneously, use `flock()`.  

```php
<?php
$file = fopen("example.txt", "a");
if (flock($file, LOCK_EX)) { // Acquire an exclusive lock
    fwrite($file, "\nLocked write operation.");
    flock($file, LOCK_UN); // Release the lock
}
fclose($file);
?>
```

---

## **10. Summary**  

| **Function** | **Purpose** |
|-------------|-------------|
| `fopen()`  | Opens a file in a specific mode. |
| `fwrite()` | Writes data to a file. |
| `fread()`  | Reads a specified number of bytes from a file. |
| `file_get_contents()` | Reads the entire file into a string. |
| `fgets()`  | Reads a single line from a file. |
| `unlink()` | Deletes a file. |
| `file_exists()` | Checks if a file exists. |
| `flock()`  | Locks a file to prevent conflicts. |
# **Sessions and Cookies in PHP**  

Sessions and cookies are essential for managing user data across multiple pages in PHP. They help track user activity, store preferences, and enable authentication mechanisms.  

---

## **1. What Are Cookies? 🍪**  

A **cookie** is a small file stored on the user's browser. It allows websites to remember user data across different visits.  

### **Setting a Cookie with `setcookie()`**  

```php
<?php
setcookie("username", "Sandeep", time() + (86400 * 7), "/"); // Cookie expires in 7 days
?>
```

🔹 **Parameters:**
- `"username"` → Name of the cookie.
- `"Sandeep"` → Value of the cookie.
- `time() + (86400 * 7)` → Expiration time (7 days).
- `"/"` → Cookie is available across the entire website.

---

### **Retrieving a Cookie**  

```php
<?php
if (isset($_COOKIE["username"])) {
    echo "Welcome back, " . $_COOKIE["username"];
} else {
    echo "Cookie not found!";
}
?>
```

---

### **Deleting a Cookie**  

To delete a cookie, set its expiration time to the past:  

```php
<?php
setcookie("username", "", time() - 3600, "/"); // Cookie expires immediately
?>
```

---

## **2. What Are Sessions? 🔐**  

A **session** stores user data on the server (unlike cookies, which store data in the browser). Sessions are useful for authentication and managing sensitive user information.  

### **Starting a Session with `session_start()`**  

Before using a session, call `session_start()` at the top of your PHP script.  

```php
<?php
session_start();  // Start a session
$_SESSION["user"] = "Sandeep"; // Store session data
echo "Session set successfully!";
?>
```

---

### **Accessing Session Data**  

```php
<?php
session_start();
echo "Hello, " . $_SESSION["user"];
?>
```

---

### **Destroying a Session**  

To log out a user and clear session data:  

```php
<?php
session_start();
session_unset();  // Unset all session variables
session_destroy(); // Destroy the session
echo "Session destroyed!";
?>
```

---

## **3. Cookies vs. Sessions: Key Differences**  

| Feature   | Cookies  | Sessions  |
|-----------|---------|-----------|
| **Storage Location** | Client-side (browser) | Server-side |
| **Data Lifetime** | Until expiration (or deleted) | Until user logs out or session times out |
| **Security** | Less secure (stored in browser) | More secure (stored on server) |
| **Best Use Case** | User preferences (e.g., "Remember Me") | User authentication (e.g., login sessions) |

---

## **4. Best Practices for Managing Sessions & Cookies**  

✅ **Use `secure` and `httponly` flags for cookies:**  
```php
setcookie("authToken", "xyz123", time() + 3600, "/", "", true, true);
```
- `true, true` ensures secure transfer over HTTPS and prevents JavaScript access.

✅ **Regenerate Session ID on Login:**  
```php
session_regenerate_id(true); // Prevents session fixation attacks
```

✅ **Set a session timeout:**  
```php
if (!isset($_SESSION["last_activity"])) {
    $_SESSION["last_activity"] = time();
} elseif (time() - $_SESSION["last_activity"] > 1800) { // 30 mins
    session_unset();
    session_destroy();
}
```
# **Handling Form Data in PHP**  

Handling form data securely is essential in PHP to prevent attacks like SQL injection, XSS, and CSRF. This guide covers `$_GET`, `$_POST`, input validation, and security best practices.  

---

## **1. Using `$_GET` and `$_POST` to Receive Data**  

PHP provides two superglobals for handling form data:  
- **`$_GET`**: Retrieves data from the URL query string.
- **`$_POST`**: Retrieves data from the request body (more secure for sensitive data).  

### **Example: Simple HTML Form**  

```html
<form action="process.php" method="post">
    Name: <input type="text" name="name">
    Email: <input type="email" name="email">
    <input type="submit" value="Submit">
</form>
```

---

### **Handling Form Data with `$_POST` (Recommended)**  

```php
<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $name = $_POST["name"];
    $email = $_POST["email"];
    echo "Name: " . $name . "<br>";
    echo "Email: " . $email;
}
?>
```

📌 **`$_POST` is preferred over `$_GET` for handling sensitive data** (e.g., passwords) because `$_GET` exposes data in the URL.

---

### **Handling Form Data with `$_GET`**  

```html
<form action="process.php" method="get">
    Search: <input type="text" name="query">
    <input type="submit" value="Search">
</form>
```

```php
<?php
if (isset($_GET["query"])) {
    echo "Search Query: " . $_GET["query"];
}
?>
```

🔹 **Use cases for `$_GET`:**
- Search queries
- Pagination (`page=2`)
- Sharing URLs with parameters

---

## **2. Input Validation in PHP**  

Validating form inputs prevents incorrect or malicious data.  

### **Basic Validation Example**  

```php
<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    if (empty($_POST["name"])) {
        echo "Name is required!";
    } elseif (!preg_match("/^[a-zA-Z ]*$/", $_POST["name"])) {
        echo "Only letters and spaces allowed!";
    } else {
        echo "Valid name: " . $_POST["name"];
    }
}
?>
```

---

## **3. Input Sanitization in PHP**  

Sanitization ensures user input is safe before using it in the database or displaying it on a webpage.  

### **Sanitize User Input**  

```php
$name = filter_var($_POST["name"], FILTER_SANITIZE_STRING);
$email = filter_var($_POST["email"], FILTER_SANITIZE_EMAIL);
```

### **Validate Email**  

```php
if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
    echo "Invalid email format!";
}
```

---

## **4. Preventing SQL Injection**  

SQL injection can be prevented using **prepared statements** with PDO or MySQLi.  

```php
$conn = new PDO("mysql:host=localhost;dbname=test", "root", "");
$stmt = $conn->prepare("INSERT INTO users (name, email) VALUES (:name, :email)");
$stmt->bindParam(":name", $name);
$stmt->bindParam(":email", $email);
$stmt->execute();
```

📌 **Never use raw SQL queries with user input!** 🚨

---

## **5. Preventing Cross-Site Scripting (XSS)**  

XSS attacks can inject malicious JavaScript. Use `htmlspecialchars()` to escape user input before outputting it.  

```php
echo htmlspecialchars($_POST["name"], ENT_QUOTES, 'UTF-8');
```

---

## **6. Protecting Against CSRF Attacks**  

CSRF (Cross-Site Request Forgery) tricks users into performing unwanted actions. A **CSRF token** prevents this.  

### **Generating a CSRF Token**  

```php
session_start();
if (empty($_SESSION['csrf_token'])) {
    $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
}
```

### **Adding CSRF Token to Forms**  

```html
<form action="process.php" method="post">
    <input type="hidden" name="csrf_token" value="<?php echo $_SESSION['csrf_token']; ?>">
    Name: <input type="text" name="name">
    <input type="submit" value="Submit">
</form>
```

### **Verifying CSRF Token in PHP**  

```php
if ($_POST['csrf_token'] !== $_SESSION['csrf_token']) {
    die("CSRF attack detected!");
}
```

---

## **7. Secure File Upload Handling**  

If your form allows file uploads, **validate file types and size**.  

```php
if (isset($_FILES["file"])) {
    $allowedTypes = ["image/png", "image/jpeg"];
    if (in_array($_FILES["file"]["type"], $allowedTypes) && $_FILES["file"]["size"] < 2000000) {
        move_uploaded_file($_FILES["file"]["tmp_name"], "uploads/" . $_FILES["file"]["name"]);
        echo "File uploaded successfully!";
    } else {
        echo "Invalid file!";
    }
}
```

---
# **PHP and MySQL Interaction: Connecting, CRUD Operations, and Secure Queries with PDO**  

PHP interacts with MySQL using two main extensions:  
1. **MySQLi** (MySQL Improved)  
2. **PDO** (PHP Data Objects) – Recommended for secure and flexible database handling  

---

## **1. Connecting PHP to MySQL**  

### **Using MySQLi (Procedural Style)**
```php
$servername = "localhost";
$username = "root";
$password = "";
$database = "testdb";

$conn = mysqli_connect($servername, $username, $password, $database);

if (!$conn) {
    die("Connection failed: " . mysqli_connect_error());
}
echo "Connected successfully!";
```

### **Using PDO (Recommended)**
```php
try {
    $conn = new PDO("mysql:host=localhost;dbname=testdb", "root", "");
    $conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    echo "Connected successfully!";
} catch (PDOException $e) {
    echo "Connection failed: " . $e->getMessage();
}
```

📌 **Why Use PDO?**
- Supports multiple databases (MySQL, PostgreSQL, SQLite, etc.)
- Provides built-in security with **prepared statements**
- Easier error handling with exceptions

---

## **2. Performing CRUD Operations in PHP with MySQL**

### **(C) Create: Insert Data into MySQL**  

#### **Using MySQLi**
```php
$sql = "INSERT INTO users (name, email) VALUES ('John Doe', 'john@example.com')";
if (mysqli_query($conn, $sql)) {
    echo "New record created successfully!";
} else {
    echo "Error: " . mysqli_error($conn);
}
```

#### **Using PDO (Prepared Statement)**
```php
$sql = "INSERT INTO users (name, email) VALUES (:name, :email)";
$stmt = $conn->prepare($sql);
$stmt->execute(["name" => "John Doe", "email" => "john@example.com"]);
echo "New record inserted!";
```

✅ **Using `prepare()` prevents SQL injection!**

---

### **(R) Read: Fetch Data from MySQL**  

#### **Using MySQLi**
```php
$result = mysqli_query($conn, "SELECT * FROM users");
while ($row = mysqli_fetch_assoc($result)) {
    echo "Name: " . $row["name"] . " - Email: " . $row["email"] . "<br>";
}
```

#### **Using PDO**
```php
$stmt = $conn->query("SELECT * FROM users");
while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
    echo "Name: " . $row["name"] . " - Email: " . $row["email"] . "<br>";
}
```

---

### **(U) Update: Modify Data in MySQL**  

#### **Using MySQLi**
```php
$sql = "UPDATE users SET email='newemail@example.com' WHERE name='John Doe'";
mysqli_query($conn, $sql);
```

#### **Using PDO**
```php
$sql = "UPDATE users SET email=:email WHERE name=:name";
$stmt = $conn->prepare($sql);
$stmt->execute(["email" => "newemail@example.com", "name" => "John Doe"]);
```

---

### **(D) Delete: Remove Data from MySQL**  

#### **Using MySQLi**
```php
$sql = "DELETE FROM users WHERE name='John Doe'";
mysqli_query($conn, $sql);
```

#### **Using PDO**
```php
$sql = "DELETE FROM users WHERE name=:name";
$stmt = $conn->prepare($sql);
$stmt->execute(["name" => "John Doe"]);
```

---

## **3. Using Prepared Statements to Prevent SQL Injection**  
🚨 **Always use prepared statements for security!**  

**❌ Vulnerable Code (SQL Injection)**
```php
$name = $_GET["name"];
$sql = "SELECT * FROM users WHERE name = '$name'";  // 🚨 Hackers can inject SQL here!
$result = mysqli_query($conn, $sql);
```

**✅ Secure Code Using PDO**
```php
$sql = "SELECT * FROM users WHERE name = :name";
$stmt = $conn->prepare($sql);
$stmt->execute(["name" => $_GET["name"]]);
```

---
# **PHP and AJAX: Making Asynchronous Requests**  

AJAX (Asynchronous JavaScript and XML) allows web applications to send and receive data from the server **without refreshing the page**. PHP is commonly used as the backend to process AJAX requests.

---

## **1. How AJAX Works in PHP**
The process follows these steps:  

1️⃣ **User triggers an event** (e.g., clicking a button)  
2️⃣ **JavaScript sends an AJAX request** using `XMLHttpRequest` or `fetch()`  
3️⃣ **PHP processes the request** and returns data  
4️⃣ **JavaScript updates the page** dynamically  

---

## **2. Example 1: Fetch Data Using AJAX and PHP**
Let’s create a simple example where we **fetch user data** from the server using AJAX.

### **📌 Step 1: Create an HTML & JavaScript File (`index.html`)**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AJAX with PHP</title>
</head>
<body>
    <h2>User Information</h2>
    <button onclick="loadUser()">Get User Info</button>
    <p id="result"></p>

    <script>
        function loadUser() {
            let xhr = new XMLHttpRequest();
            xhr.open("GET", "fetch_user.php", true);
            xhr.onreadystatechange = function () {
                if (xhr.readyState == 4 && xhr.status == 200) {
                    document.getElementById("result").innerHTML = xhr.responseText;
                }
            };
            xhr.send();
        }
    </script>
</body>
</html>
```

---

### **📌 Step 2: Create a PHP File (`fetch_user.php`)**
```php
<?php
// Simulating fetching data from a database
$user = [
    "name" => "John Doe",
    "email" => "john@example.com",
    "age" => 25
];

// Convert PHP array to JSON
echo json_encode($user);
?>
```

---

### **📌 Step 3: Improve the AJAX Request Using `fetch()`**
We can also use `fetch()` instead of `XMLHttpRequest`:

```html
<script>
    function loadUser() {
        fetch("fetch_user.php")
            .then(response => response.json())
            .then(data => {
                document.getElementById("result").innerHTML = 
                    `Name: ${data.name} <br> Email: ${data.email} <br> Age: ${data.age}`;
            })
            .catch(error => console.log("Error: " + error));
    }
</script>
```

---

## **3. Example 2: Submitting Form Data Using AJAX**
Let’s send a **POST request** to the server with form data.

### **📌 Step 1: Create a Form (`form.html`)**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AJAX Form</title>
</head>
<body>
    <h2>Submit Data Using AJAX</h2>
    <form id="userForm">
        <input type="text" id="name" placeholder="Enter your name" required><br><br>
        <input type="email" id="email" placeholder="Enter your email" required><br><br>
        <button type="submit">Submit</button>
    </form>
    <p id="response"></p>

    <script>
        document.getElementById("userForm").addEventListener("submit", function(event) {
            event.preventDefault();

            let formData = new FormData();
            formData.append("name", document.getElementById("name").value);
            formData.append("email", document.getElementById("email").value);

            fetch("process_form.php", {
                method: "POST",
                body: formData
            })
            .then(response => response.text())
            .then(data => {
                document.getElementById("response").innerHTML = data;
            })
            .catch(error => console.error("Error:", error));
        });
    </script>
</body>
</html>
```

---

### **📌 Step 2: Process the Form Data (`process_form.php`)**
```php
<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $name = htmlspecialchars($_POST["name"]);
    $email = htmlspecialchars($_POST["email"]);

    // Simulating saving data to a database
    echo "Data Received: <br> Name: $name <br> Email: $email";
}
?>
```

---
# **PHP Security: Common Vulnerabilities and Best Practices**  

PHP applications are vulnerable to several security threats if not properly secured. Understanding these vulnerabilities and implementing best practices is crucial to safeguarding your web applications.  

---

## **1. SQL Injection**  
SQL Injection (SQLi) occurs when an attacker manipulates SQL queries by injecting malicious input into an application’s database query.  

### **🚨 Example of Vulnerable Code**  
```php
$username = $_GET['username'];
$query = "SELECT * FROM users WHERE username = '$username'";  
$result = mysqli_query($conn, $query);
```
👉 **Problem**: If an attacker inputs `admin' --`, the SQL query will break, potentially exposing user data.  

### **✅ Prevention: Use Prepared Statements**  
```php
$stmt = $conn->prepare("SELECT * FROM users WHERE username = ?");
$stmt->bind_param("s", $username);
$stmt->execute();
```
🔹 **Use PDO or MySQLi prepared statements** to prevent SQL injection.  

---

## **2. Cross-Site Scripting (XSS)**  
XSS allows attackers to inject malicious JavaScript into web pages, affecting users who visit the page.  

### **🚨 Example of Vulnerable Code**  
```php
echo "Welcome, " . $_GET['name'];  
```
👉 If an attacker submits `<script>alert('Hacked!')</script>`, it will execute JavaScript in the user’s browser.  

### **✅ Prevention: Escape Output Properly**  
```php
echo "Welcome, " . htmlspecialchars($_GET['name'], ENT_QUOTES, 'UTF-8');  
```
🔹 **Use `htmlspecialchars()` to encode special characters** and prevent script execution.  

---

## **3. Cross-Site Request Forgery (CSRF)**  
CSRF tricks authenticated users into executing unwanted actions, such as changing their passwords or making transactions.  

### **🚨 Example of Vulnerable Code**  
A user is logged in, and an attacker sends a malicious link:  
```html
<img src="http://example.com/change_password.php?new_pass=hacked123">
```
If the user is logged in, the request will be executed **without their consent**.  

### **✅ Prevention: Use CSRF Tokens**  
```php
session_start();
$_SESSION['csrf_token'] = bin2hex(random_bytes(32));
```
Include this CSRF token in your forms and verify it before processing requests.  

```php
if ($_POST['csrf_token'] !== $_SESSION['csrf_token']) {
    die("CSRF attack detected!");
}
```

🔹 **Always validate user actions with CSRF tokens.**  

---

## **4. Remote File Inclusion (RFI) & Local File Inclusion (LFI)**  
Attackers can exploit improperly handled file includes to execute malicious code.  

### **🚨 Example of Vulnerable Code**  
```php
$page = $_GET['page'];
include($page . ".php");
```
👉 If an attacker sends `?page=../../etc/passwd`, they might read system files!  

### **✅ Prevention: Restrict File Access**  
```php
$allowed_pages = ['home', 'about', 'contact'];
if (in_array($page, $allowed_pages)) {
    include($page . ".php");
} else {
    die("Access denied!");
}
```
🔹 **Never accept user input directly for file inclusion.**  

---

## **5. Best Practices for Secure PHP Coding**  
✅ **Sanitize User Input:** Use `filter_var()` and `htmlspecialchars()`.  
✅ **Use Prepared Statements:** Prevent SQL injection with `PDO` or `MySQLi`.  
✅ **Validate Data:** Ensure input is of the correct type before processing.  
✅ **Implement HTTPS:** Encrypt sensitive data transmissions.  
✅ **Use Secure Session Management:** Regenerate session IDs and set `HttpOnly` cookies.  
✅ **Restrict File Uploads:** Validate file types and store them securely.  
✅ **Monitor Logs:** Keep track of unusual activities in logs.  
# **Composer: PHP Dependency Management**  

Composer is a dependency manager for PHP, allowing developers to manage libraries and packages efficiently. It simplifies package installation, updates, and autoloading in PHP projects.  

---

## **1. What is Composer?**  
Composer is a tool that:  
✅ Installs and updates third-party PHP packages.  
✅ Resolves dependencies automatically.  
✅ Uses **Packagist** (the main repository for PHP packages).  
✅ Provides **autoloading** to avoid manual `require` statements.  

🔹 **Installation:** Download Composer from [getcomposer.org](https://getcomposer.org).  

```bash
php -r "copy('https://getcomposer.org/installer', 'composer-setup.php');"
php composer-setup.php
php -r "unlink('composer-setup.php');"
```

Verify installation:  
```bash
composer --version
```

---

## **2. `composer.json` – Defining Dependencies**  
The `composer.json` file stores package details and project metadata.  

### **🚀 Example: A Simple `composer.json` File**  
```json
{
  "name": "myproject/app",
  "description": "A sample PHP project",
  "require": {
    "monolog/monolog": "^3.0"
  },
  "autoload": {
    "psr-4": {
      "App\\": "src/"
    }
  }
}
```
🔹 **Key Sections:**  
- `"require"`: Lists dependencies (e.g., `monolog/monolog`).  
- `"autoload"`: Defines PSR-4 autoloading for classes.  

---

## **3. Installing Dependencies**  
After defining dependencies, install them with:  
```bash
composer install
```
This creates a `vendor/` directory and generates a `composer.lock` file.  

To add new packages:  
```bash
composer require guzzlehttp/guzzle
```
This automatically updates `composer.json` and downloads the package.  

To update all dependencies:  
```bash
composer update
```

---

## **4. Autoloading Classes**  
Composer provides **automatic class loading**, eliminating the need for manual `require` statements.  

### **🚀 Example: Using Autoloading**
1️⃣ Define a class in `src/Greeting.php`:  
```php
namespace App;

class Greeting {
    public function sayHello() {
        return "Hello, Composer!";
    }
}
```
  
2️⃣ Include the autoloader in `index.php`:  
```php
require 'vendor/autoload.php';

use App\Greeting;

$greet = new Greeting();
echo $greet->sayHello();
```
Now, running `php index.php` outputs:  
```
Hello, Composer!
```

🔹 **Why Use Composer's Autoload?**  
✔️ Automatically loads classes based on namespaces.  
✔️ Supports **PSR-4 autoloading** for structured projects.  
✔️ Reduces unnecessary `require` statements.  

---

## **5. Useful Composer Commands**  
🔹 **Check outdated packages:**  
```bash
composer outdated
```
🔹 **Remove a package:**  
```bash
composer remove monolog/monolog
```
🔹 **Dump autoload (when adding new classes manually):**  
```bash
composer dump-autoload
```

---
# **PHP Frameworks: Overview & Comparison**  

### **1. What is a PHP Framework?**  
A **PHP framework** is a collection of pre-written code that provides a structured way to build web applications. It helps developers:  
✅ Write cleaner and maintainable code.  
✅ Follow **MVC (Model-View-Controller)** architecture.  
✅ Improve security and performance.  
✅ Reduce development time with built-in features.  

---

### **2. Why Use a PHP Framework?**  
🔹 **Faster Development:** Pre-built components reduce coding effort.  
🔹 **Security:** Built-in protection against SQL injection, XSS, and CSRF.  
🔹 **Scalability:** Easier to maintain large applications.  
🔹 **Built-in Features:** Routing, templating, authentication, and ORM.  
🔹 **Community Support:** Well-documented and frequently updated.  

---

### **3. Popular PHP Frameworks: A Comparison**  

| Feature        | **Laravel** 🚀 | **Symfony** 🏗️ | **CodeIgniter** ⚡ |
|--------------|--------------|--------------|--------------|
| **Ease of Use** | ✅ Beginner-friendly | ❌ Steep learning curve | ✅ Simple & lightweight |
| **Performance** | ⚡ Moderate | 🚀 High-performance | ⚡ Fast |
| **Architecture** | MVC + Blade Templating | Component-based + MVC | MVC |
| **Database ORM** | ✅ Eloquent | ✅ Doctrine | ❌ Query Builder |
| **Security Features** | ✅ CSRF, XSS, Encryption | ✅ Advanced Security | ✅ Basic Security |
| **Built-in Features** | ✅ Authentication, Queue, Mail | ✅ Highly customizable | ❌ Minimal features |
| **Community Support** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

---

### **4. Framework Breakdown**  

#### **🔹 Laravel (Best for Rapid Development & Startups)**
✅ **Why Choose Laravel?**  
- Elegant **Eloquent ORM** for database handling.  
- **Blade templating engine** for clean views.  
- **Built-in authentication** and security features.  
- **Artisan CLI** for automation.  
- **RESTful routing** for API development.  

✅ **Best For:**  
- Web applications & APIs.  
- Startups and enterprise apps.  
- Developers who prefer a smooth learning curve.  

---

#### **🔹 Symfony (Best for Enterprise-Level Applications)**
✅ **Why Choose Symfony?**  
- Highly **modular** and **scalable**.  
- Uses **Doctrine ORM** for advanced database management.  
- **Twig templating** for clean UI rendering.  
- Robust **security features** (firewalls, encryption, OAuth).  
- Preferred for large, complex applications.  

✅ **Best For:**  
- Enterprise software.  
- Large-scale web applications.  
- Developers needing full flexibility.  

---

#### **🔹 CodeIgniter (Best for Small, Fast Applications)**
✅ **Why Choose CodeIgniter?**  
- **Lightweight & fast** (great for shared hosting).  
- No complex dependencies.  
- Uses **query builder** instead of full ORM.  
- Minimal configuration required.  

✅ **Best For:**  
- Small projects & microservices.  
- Developers who want a simple, **no-fuss** framework.  
- Apps requiring **high performance** with minimal overhead.  

---
Building a **REST API in PHP** involves setting up routing, handling requests, implementing authentication, and formatting responses in JSON. Here's how you can do it step by step:

---

## 1. **Setting Up Routing**
Routing in a PHP REST API is responsible for mapping incoming requests (e.g., `GET /users`, `POST /users`) to specific functions.

### Basic Routing Example:
```php
$requestMethod = $_SERVER["REQUEST_METHOD"];
$requestUri = explode("/", trim($_SERVER["REQUEST_URI"], "/"));

// Example: If request is GET /users
if ($requestMethod == "GET" && $requestUri[0] == "users") {
    getUsers();
} elseif ($requestMethod == "POST" && $requestUri[0] == "users") {
    createUser();
} else {
    response(404, ["message" => "Not Found"]);
}
```

---

## 2. **Handling Requests**
You need to read input data and process it based on the request method.

### Example: Handling GET and POST requests
```php
function getUsers() {
    // Fetch users from a database (example array for simplicity)
    $users = [
        ["id" => 1, "name" => "Alice"],
        ["id" => 2, "name" => "Bob"]
    ];
    
    response(200, $users);
}

function createUser() {
    $inputData = json_decode(file_get_contents("php://input"), true);
    
    if (!isset($inputData["name"])) {
        response(400, ["error" => "Name is required"]);
    }
    
    // Here you would insert into a database
    $newUser = ["id" => rand(3, 100), "name" => $inputData["name"]];
    
    response(201, $newUser);
}
```

---

## 3. **Implementing Authentication**
A REST API usually requires authentication, such as **JWT (JSON Web Token)** or an **API key**.

### Example: Simple Token Authentication
```php
function authenticate() {
    $headers = apache_request_headers();
    
    if (!isset($headers["Authorization"])) {
        response(401, ["error" => "Unauthorized"]);
    }

    $token = $headers["Authorization"];
    
    // Validate token (For simplicity, checking against a static token)
    if ($token !== "Bearer my_secret_token") {
        response(403, ["error" => "Forbidden"]);
    }
}
```
#### **Usage**
Call `authenticate();` at the start of protected endpoints.

---

## 4. **Formatting Responses in JSON**
A REST API should return JSON responses with appropriate HTTP status codes.

### Example: JSON Response Function
```php
function response($statusCode, $data) {
    header("Content-Type: application/json");
    http_response_code($statusCode);
    echo json_encode($data);
    exit();
}
```
#### Example Responses:
- **Success:** `response(200, ["message" => "Success"]);`
- **Not Found:** `response(404, ["error" => "Resource not found"]);`

---

## 5. **Handling Errors**
Proper error handling ensures users get clear error messages.

### Example: Global Error Handling
```php
set_exception_handler(function ($e) {
    response(500, ["error" => $e->getMessage()]);
});
```

---

## **Final Example: A Simple PHP REST API**
```php
<?php
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type, Authorization");

// Basic Routing
$requestMethod = $_SERVER["REQUEST_METHOD"];
$requestUri = explode("/", trim($_SERVER["REQUEST_URI"], "/"));

function getUsers() {
    $users = [
        ["id" => 1, "name" => "Alice"],
        ["id" => 2, "name" => "Bob"]
    ];
    response(200, $users);
}

function createUser() {
    $inputData = json_decode(file_get_contents("php://input"), true);
    if (!isset($inputData["name"])) {
        response(400, ["error" => "Name is required"]);
    }
    $newUser = ["id" => rand(3, 100), "name" => $inputData["name"]];
    response(201, $newUser);
}

// Routing Logic
if ($requestMethod == "GET" && $requestUri[0] == "users") {
    getUsers();
} elseif ($requestMethod == "POST" && $requestUri[0] == "users") {
    createUser();
} else {
    response(404, ["error" => "Not Found"]);
}

// JSON Response Function
function response($statusCode, $data) {
    header("Content-Type: application/json");
    http_response_code($statusCode);
    echo json_encode($data);
    exit();
}
?>
```

---

## **Conclusion**
✅ **Routing**: Use `$_SERVER["REQUEST_METHOD"]` and `$_SERVER["REQUEST_URI"]`.  
✅ **Handling Requests**: Use `file_get_contents("php://input")` for JSON input.  
✅ **Authentication**: Validate API keys or JWT tokens.  
✅ **Response Formatting**: Always return JSON responses with correct HTTP status codes.

## **Using PHP for Command-Line Scripting**
PHP isn't just for web development—it can also be used for **CLI (Command-Line Interface) scripting** to automate tasks like data processing, system administration, and background jobs.

---

## **1. Executing PHP Scripts via CLI**
To run a PHP script in the terminal:

### **Steps to Execute a PHP Script**
1. **Create a PHP script**  
   Example: `script.php`
   ```php
   <?php
   echo "Hello, CLI!\n";
   ```
2. **Run it from the terminal**  
   ```sh
   php script.php
   ```
   **Output:**  
   ```
   Hello, CLI!
   ```

### **Make the Script Executable (Linux/macOS)**
1. Add the shebang (`#!`) at the top:
   ```php
   #!/usr/bin/php
   <?php
   echo "Hello, CLI!\n";
   ```
2. Make it executable:
   ```sh
   chmod +x script.php
   ```
3. Run it:
   ```sh
   ./script.php
   ```

---

## **2. Handling Command-Line Arguments**
Command-line arguments can be accessed via the `$argv` and `$argc` variables.

### **Example: Accepting User Input**
Create `cli_args.php`:
```php
<?php
if ($argc < 2) {
    echo "Usage: php cli_args.php <your_name>\n";
    exit(1);
}

$name = $argv[1];
echo "Hello, $name!\n";
```
**Run it:**
```sh
php cli_args.php Sandeep
```
**Output:**
```
Hello, Sandeep!
```
- `$argv`: An array containing arguments (`$argv[0]` is the script name).
- `$argc`: The number of arguments.

---

## **3. Reading User Input in Real-Time**
You can prompt users for input dynamically using `readline()`.

### **Example: Prompting for User Input**
```php
<?php
echo "Enter your name: ";
$name = trim(fgets(STDIN));
echo "Hello, $name!\n";
```
**Run:**
```sh
php script.php
```
**Output:**
```
Enter your name: Sandeep
Hello, Sandeep!
```

---

## **4. Automating Tasks with PHP CLI**
PHP can be used for cron jobs, backups, and automation scripts.

### **Example: Automating File Cleanup**
Create `cleanup.php`:
```php
<?php
$dir = "/path/to/logs";
$files = glob("$dir/*.log");

foreach ($files as $file) {
    if (filemtime($file) < time() - 7 * 86400) { // Older than 7 days
        unlink($file);
        echo "Deleted: $file\n";
    }
}
```
**Schedule a Cron Job (Linux/macOS)**
```sh
crontab -e
```
Add a line:
```
0 3 * * * php /path/to/cleanup.php
```
This runs the script **daily at 3 AM**.

---

## **5. Working with Environment Variables**
Environment variables help in passing configuration values.

### **Example: Using `getenv()`**
```php
<?php
$apiKey = getenv("API_KEY") ?: "default_key";
echo "API Key: $apiKey\n";
```
**Set and run:**
```sh
export API_KEY="my_secret_key"
php script.php
```
**Output:**
```
API Key: my_secret_key
```

---

## **6. Running Background Jobs**
Run PHP scripts in the background using `nohup` or `&`.

```sh
nohup php long_task.php > output.log 2>&1 &
```
- `nohup`: Runs even after logout.
- `&`: Runs in the background.

---

## **7. Using PHP CLI for Database Operations**
You can use PHP CLI to interact with a database.

### **Example: Fetch Data from MySQL**
```php
<?php
$pdo = new PDO("mysql:host=localhost;dbname=mydb", "user", "pass");

$stmt = $pdo->query("SELECT name FROM users");
while ($row = $stmt->fetch()) {
    echo $row['name'] . "\n";
}
```
Run:
```sh
php fetch_users.php
```

---

## **Conclusion**
✅ **Execute PHP scripts via CLI** using `php script.php`  
✅ **Handle arguments** using `$argv` and `$argc`  
✅ **Read user input** using `fgets(STDIN)`  
✅ **Automate tasks** using **cron jobs**  
✅ **Work with environment variables** using `getenv()`  
✅ **Run background jobs** with `nohup`  
Optimizing PHP performance is crucial for building fast, scalable, and efficient applications. Here are the best practices, focusing on caching, opcode caching (`OPcache`), and database query optimization:

---

### **1. Caching for Improved Performance**
Caching helps reduce redundant processing and speeds up data retrieval. There are different types of caching strategies:

#### **a) Page Caching**
- Stores entire HTML responses to serve them quickly.
- Tools: **Varnish, Nginx FastCGI Cache, WordPress WP Super Cache.**

#### **b) Object Caching**
- Stores frequently used objects in memory.
- Tools: **Memcached, Redis.**
- Example:
  ```php
  $redis = new Redis();
  $redis->connect('127.0.0.1', 6379);
  $redis->set("username", "Sandeep");
  echo $redis->get("username"); // Outputs: Sandeep
  ```

#### **c) HTTP Caching**
- Uses `ETag`, `Last-Modified`, `Cache-Control` headers to reduce repeated HTTP requests.

#### **d) Data Caching**
- Saves frequently accessed data in memory to avoid repeated database calls.
- Example:
  ```php
  $cacheKey = "products_list";
  if (!$data = $redis->get($cacheKey)) {
      $data = getProductsFromDatabase(); // Expensive DB Query
      $redis->setex($cacheKey, 3600, json_encode($data)); // Store in cache for 1 hour
  }
  ```

---

### **2. Opcode Caching with `OPcache`**
`OPcache` is a built-in PHP extension that speeds up execution by storing compiled bytecode in memory, reducing script compilation overhead.

#### **How to Enable OPcache**
- Ensure `opcache` is enabled in `php.ini`:
  ```ini
  opcache.enable=1
  opcache.memory_consumption=128
  opcache.max_accelerated_files=4000
  opcache.validate_timestamps=1
  ```
- Verify if `OPcache` is enabled:
  ```php
  phpinfo();
  ```
- Use `opcache_reset()` wisely when deploying updates to avoid stale cache issues.

---

### **3. Database Query Optimization**
Optimizing SQL queries is crucial for performance, especially for high-traffic applications.

#### **a) Use Proper Indexing**
- Index frequently queried columns.
- Use `EXPLAIN` to analyze queries:
  ```sql
  EXPLAIN SELECT * FROM users WHERE email = 'test@example.com';
  ```
- Example Index:
  ```sql
  CREATE INDEX idx_email ON users(email);
  ```

#### **b) Use `LIMIT` and `OFFSET` for Pagination**
Instead of:
```sql
SELECT * FROM products;
```
Use:
```sql
SELECT * FROM products LIMIT 20 OFFSET 40;
```

#### **c) Avoid `SELECT *`**
Specify only required columns:
```sql
SELECT id, name FROM users;
```

#### **d) Use Prepared Statements**
Prevents SQL injection and improves query performance:
```php
$stmt = $pdo->prepare("SELECT * FROM users WHERE email = ?");
$stmt->execute([$email]);
$user = $stmt->fetch();
```

#### **e) Optimize Joins & Use Proper Data Types**
- Use indexed columns for `JOIN` conditions.
- Select minimal data needed for operations.

#### **f) Use Query Caching**
- Store frequently executed queries in **Redis** or **Memcached**.

---

### **Conclusion**
1. **Implement Caching** (Page, Object, HTTP, Data) to minimize redundant processing.
2. **Enable OPcache** to eliminate repeated script compilation.
3. **Optimize Database Queries** by indexing, limiting results, and using efficient queries.
Testing and debugging are crucial for developing stable and efficient PHP applications. Let’s explore key debugging tools and testing methodologies:

---

## **1. Debugging PHP Applications**
Effective debugging helps identify and fix errors efficiently. Here are some common debugging techniques:

### **a) Using `var_dump()` for Quick Debugging**
- `var_dump()` is a simple but effective way to inspect variables.
- Example:
  ```php
  $user = ["name" => "Sandeep", "age" => 25];
  var_dump($user);
  ```
  Output:
  ```
  array(2) {
    ["name"]=> string(7) "Sandeep"
    ["age"]=> int(25)
  }
  ```

- Alternative: `print_r()` for a cleaner output:
  ```php
  print_r($user);
  ```
  Output:
  ```
  Array ( [name] => Sandeep [age] => 25 )
  ```

- For better readability, wrap debugging output in `<pre>`:
  ```php
  echo "<pre>";
  var_dump($user);
  echo "</pre>";
  ```

---

### **b) Using `xdebug` for Advanced Debugging**
`Xdebug` is a powerful tool that provides:
- Stack traces
- Code coverage analysis
- Step-by-step debugging

#### **Installing `Xdebug`**
1. Install `Xdebug`:
   ```bash
   sudo apt-get install php-xdebug  # Ubuntu/Debian
   brew install xdebug  # macOS (Homebrew)
   ```
2. Enable it in `php.ini`:
   ```ini
   zend_extension=xdebug.so
   xdebug.mode=debug
   xdebug.start_with_request=yes
   xdebug.client_host=127.0.0.1
   xdebug.client_port=9003
   ```
3. Restart Apache/Nginx:
   ```bash
   sudo systemctl restart apache2  # Apache
   sudo systemctl restart nginx    # Nginx
   ```

#### **Using `Xdebug` with VS Code**
- Install **PHP Debug** extension in VS Code.
- Add a debug configuration in `.vscode/launch.json`:
  ```json
  {
      "version": "0.2.0",
      "configurations": [
          {
              "name": "Listen for Xdebug",
              "type": "php",
              "request": "launch",
              "port": 9003
          }
      ]
  }
  ```
- Set breakpoints in VS Code and start debugging.

---

### **c) Debugging Best Practices**
- **Use Logging Instead of `var_dump()` in Production**:
  - Use **Monolog**:
    ```php
    use Monolog\Logger;
    use Monolog\Handler\StreamHandler;

    $log = new Logger('app');
    $log->pushHandler(new StreamHandler('app.log', Logger::WARNING));

    $log->warning('This is a warning message');
    ```

- **Enable Error Reporting for Development**:
  ```php
  error_reporting(E_ALL);
  ini_set('display_errors', 1);
  ```
  Disable error display in production but log errors instead:
  ```php
  ini_set('display_errors', 0);
  ini_set('log_errors', 1);
  ini_set('error_log', 'errors.log');
  ```

- **Use Exception Handling**:
  ```php
  try {
      $result = 10 / 0;
  } catch (Exception $e) {
      error_log($e->getMessage());
  }
  ```

---

## **2. Testing PHP Applications**
Testing ensures your application functions correctly and helps prevent regressions.

### **a) Unit Testing with PHPUnit**
PHPUnit is the most popular PHP testing framework for unit tests.

#### **Installing PHPUnit**
1. Install via Composer:
   ```bash
   composer require --dev phpunit/phpunit
   ```
2. Verify installation:
   ```bash
   vendor/bin/phpunit --version
   ```

#### **Writing a PHPUnit Test**
Example test for a `Calculator` class:

- **Calculator.php**
  ```php
  class Calculator {
      public function add($a, $b) {
          return $a + $b;
      }
  }
  ```
- **CalculatorTest.php**
  ```php
  use PHPUnit\Framework\TestCase;

  class CalculatorTest extends TestCase {
      public function testAddition() {
          $calc = new Calculator();
          $this->assertEquals(5, $calc->add(2, 3));
      }
  }
  ```
- Run the test:
  ```bash
  vendor/bin/phpunit tests
  ```

### **b) Functional and Integration Testing**
- Use **Laravel’s built-in testing suite** (if using Laravel).
- Use **Codeception** for end-to-end testing.
- Use **Behat** for behavior-driven testing (BDD).

---

## **Conclusion**
1. Use `var_dump()` and `print_r()` for quick debugging.
2. Use `xdebug` for advanced debugging and step-through execution.
3. Follow best practices like logging errors instead of displaying them.
4. Use PHPUnit for unit testing to ensure application reliability.
