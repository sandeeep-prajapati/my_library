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

---

## **Conclusion**  
- **Conditionals** (`if`, `else`, `switch`) allow decision-making.  
- **Loops** (`for`, `while`, `foreach`) enable repeated execution of code blocks.  

Would you like some practice problems? 🚀