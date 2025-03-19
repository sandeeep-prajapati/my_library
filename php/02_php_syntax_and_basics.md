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

---

### **Conclusion**  
- PHP is simple and flexible, making it beginner-friendly.  
- Understanding variables, data types, constants, and comments is essential for writing PHP scripts.  

Would you like some exercises to practice? 🚀