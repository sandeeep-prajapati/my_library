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

## **4. Best Practices for PHP-MySQL Interaction**  
✅ **Use PDO for flexibility and security**  
✅ **Always use prepared statements to prevent SQL injection**  
✅ **Handle errors with `try-catch` blocks in PDO**  
✅ **Close the database connection using `null` for PDO or `mysqli_close($conn)`**  
✅ **Sanitize user input before querying**  

Would you like a **full login system tutorial using PHP and MySQL**? 🚀