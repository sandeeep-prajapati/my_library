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

Want to integrate CLI with **Laravel Artisan commands** or **Symfony Console**? Let me know! 🚀