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

## **Summary: Best Practices for Secure Form Handling**  

✅ **Use `$_POST` for sensitive data.**  
✅ **Validate and sanitize all user inputs.**  
✅ **Use prepared statements for database queries.**  
✅ **Escape output to prevent XSS (`htmlspecialchars()`).**  
✅ **Implement CSRF protection using tokens.**  
✅ **Limit file upload types and sizes.**  

Would you like a **full login system demo** using PHP forms, sessions, and CSRF protection? 🚀