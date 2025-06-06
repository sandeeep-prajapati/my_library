### Notes on Integrating MySQL with PHP for Web Applications

Integrating MySQL with PHP is a common approach for developing dynamic web applications. This allows you to store, retrieve, and manipulate data in a database. Below is a guide on how to set up and use MySQL with PHP.

---

#### 1. **Prerequisites**

- **Web Server**: You need a web server like Apache or Nginx. You can use a local development environment like XAMPP, WAMP, or MAMP.

- **PHP Installation**: Ensure that you have PHP installed (preferably version 7.x or above).

- **MySQL Installation**: Make sure you have MySQL installed and running.

---

#### 2. **Connecting to MySQL Database**

You can connect to a MySQL database using the `mysqli` or `PDO` extension in PHP. Here’s how to do it with both:

**Using MySQLi:**

```php
<?php
$host = 'localhost';
$db = 'your_database';
$user = 'your_username';
$pass = 'your_password';

// Create a connection
$connection = new mysqli($host, $user, $pass, $db);

// Check the connection
if ($connection->connect_error) {
    die("Connection failed: " . $connection->connect_error);
}
echo "Connected successfully";
?>
```

**Using PDO:**

```php
<?php
$host = 'localhost';
$db = 'your_database';
$user = 'your_username';
$pass = 'your_password';

try {
    // Create a PDO instance
    $dsn = "mysql:host=$host;dbname=$db";
    $pdo = new PDO($dsn, $user, $pass);
    
    // Set the PDO error mode to exception
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    echo "Connected successfully";
} catch (PDOException $e) {
    echo "Connection failed: " . $e->getMessage();
}
?>
```

---

#### 3. **Creating a Table**

You can create a new table using SQL commands. Here’s how to execute it:

**Using MySQLi:**

```php
$create_table_sql = "CREATE TABLE IF NOT EXISTS employees (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    salary DECIMAL(10, 2) NOT NULL
)";

if ($connection->query($create_table_sql) === TRUE) {
    echo "Table created successfully";
} else {
    echo "Error creating table: " . $connection->error;
}
```

**Using PDO:**

```php
$create_table_sql = "CREATE TABLE IF NOT EXISTS employees (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    salary DECIMAL(10, 2) NOT NULL
)";

$pdo->exec($create_table_sql);
echo "Table created successfully";
```

---

#### 4. **Inserting Data**

You can insert data into the database using prepared statements to prevent SQL injection.

**Using MySQLi:**

```php
$stmt = $connection->prepare("INSERT INTO employees (name, salary) VALUES (?, ?)");
$stmt->bind_param("sd", $name, $salary);

$name = "John Doe";
$salary = 75000.00;

$stmt->execute();
$stmt->close();
echo "Data inserted successfully";
```

**Using PDO:**

```php
$stmt = $pdo->prepare("INSERT INTO employees (name, salary) VALUES (:name, :salary)");
$stmt->bindParam(':name', $name);
$stmt->bindParam(':salary', $salary);

$name = "John Doe";
$salary = 75000.00;

$stmt->execute();
echo "Data inserted successfully";
```

---

#### 5. **Selecting Data**

To retrieve data from a table, use the `SELECT` statement:

**Using MySQLi:**

```php
$result = $connection->query("SELECT * FROM employees");
while ($row = $result->fetch_assoc()) {
    echo "ID: " . $row['id'] . " - Name: " . $row['name'] . " - Salary: " . $row['salary'] . "<br>";
}
```

**Using PDO:**

```php
$stmt = $pdo->query("SELECT * FROM employees");
while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
    echo "ID: " . $row['id'] . " - Name: " . $row['name'] . " - Salary: " . $row['salary'] . "<br>";
}
```

---

#### 6. **Updating Data**

To update existing records:

**Using MySQLi:**

```php
$stmt = $connection->prepare("UPDATE employees SET salary = ? WHERE name = ?");
$stmt->bind_param("ds", $salary, $name);

$salary = 80000.00;
$name = "John Doe";

$stmt->execute();
$stmt->close();
echo "Data updated successfully";
```

**Using PDO:**

```php
$stmt = $pdo->prepare("UPDATE employees SET salary = :salary WHERE name = :name");
$stmt->bindParam(':salary', $salary);
$stmt->bindParam(':name', $name);

$salary = 80000.00;
$name = "John Doe";

$stmt->execute();
echo "Data updated successfully";
```

---

#### 7. **Deleting Data**

To delete records:

**Using MySQLi:**

```php
$stmt = $connection->prepare("DELETE FROM employees WHERE name = ?");
$stmt->bind_param("s", $name);

$name = "John Doe";
$stmt->execute();
$stmt->close();
echo "Data deleted successfully";
```

**Using PDO:**

```php
$stmt = $pdo->prepare("DELETE FROM employees WHERE name = :name");
$stmt->bindParam(':name', $name);

$name = "John Doe";
$stmt->execute();
echo "Data deleted successfully";
```

---

#### 8. **Closing the Connection**

Always remember to close your database connection after completing operations:

**Using MySQLi:**

```php
$connection->close();
```

**Using PDO:**

```php
$pdo = null; // Automatically closes the connection
```

---

### Conclusion

Integrating MySQL with PHP enables you to build powerful and dynamic web applications. By using either MySQLi or PDO, you can connect to your database and perform various operations safely and efficiently. Make sure to use prepared statements to prevent SQL injection and to manage your database connections properly to ensure the security and stability of your application.