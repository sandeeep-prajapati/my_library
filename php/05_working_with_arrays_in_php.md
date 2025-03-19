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

---

## **Conclusion**  
- **Indexed Arrays** use numeric keys.  
- **Associative Arrays** use named keys.  
- **Multidimensional Arrays** store nested arrays.  
- PHP provides **built-in functions** for array manipulation.  

Would you like some **array challenges**? 🚀