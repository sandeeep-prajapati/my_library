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

Would you like **practice exercises** on PHP file handling? 🚀