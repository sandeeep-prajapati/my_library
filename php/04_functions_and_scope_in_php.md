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

---

## **Conclusion**  
- Functions **encapsulate logic** and make code reusable.  
- They can accept **parameters** and **return values**.  
- Understanding **scope** (`local`, `global`, `static`) is crucial for variable handling.  

Would you like exercises on function handling? 🚀