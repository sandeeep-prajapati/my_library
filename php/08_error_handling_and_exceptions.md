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

## **7. Summary**  

| **Function/Concept** | **Usage** |
|----------------|-------------|
| `try-catch-finally` | Handles exceptions and ensures code execution. |
| `throw` | Throws a custom exception. |
| `set_error_handler()` | Defines a custom error handler. |
| `error_reporting()` | Sets which errors should be reported. |
| `error_log()` | Logs errors to a file. |

Would you like some **practice exercises**? 🚀