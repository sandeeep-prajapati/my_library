# **How to Debug Errors and Use Logging Features in CodeIgniter?**  

Debugging and logging are crucial for identifying and fixing issues in CodeIgniter applications. This guide will cover **error handling, debugging techniques, and logging mechanisms** in CodeIgniter.

---

## **1. Enable Debug Mode in CodeIgniter**  
To display errors, update the **environment configuration** in `env` or `public/index.php`:

📁 **`public/index.php`**
```php
define('CI_ENVIRONMENT', 'development');
```
✅ This ensures errors are displayed in the browser during development.

---

## **2. Display PHP Errors**  
Ensure error reporting is enabled by adding the following in **`app/Config/Boot/production.php`**:  
```php
error_reporting(E_ALL);
ini_set('display_errors', 1);
```
✅ This helps in **catching warnings and fatal errors**.

---

## **3. Use Debugging Tools**  
CodeIgniter provides built-in functions for debugging.

### **`var_dump()` & `print_r()` for Quick Debugging**
```php
$data = ['name' => 'John', 'email' => 'john@example.com'];
print_r($data);
```
or  
```php
var_dump($data);
```

### **CodeIgniter’s `debug()` Method**
```php
log_message('debug', 'This is a debug message');
```

---

## **4. Enable Logging in CodeIgniter**  
Logging helps track errors and issues in production without displaying them.

### **Configure Logging Settings**  
Modify **`app/Config/Logger.php`**:
```php
public $threshold = 4; // Log all levels (1-4)
```

| Threshold Level | Log Type |
|---------------|-------------|
| 0 | No logging |
| 1 | Errors only |
| 2 | Warnings & Errors |
| 3 | Info, Warnings, & Errors |
| 4 | Debug, Info, Warnings, & Errors (All logs) |

---

## **5. Writing Custom Log Messages**  
Use `log_message()` to record logs.

### **Log an Error**
```php
log_message('error', 'Database connection failed');
```

### **Log an Info Message**
```php
log_message('info', 'User logged in successfully');
```

### **Log a Debug Message**
```php
log_message('debug', 'Executing function getUserData()');
```

📁 **Log files are stored in `writable/logs/`**.

---

## **6. Handling Exceptions with Try-Catch**
Use `try-catch` blocks to handle exceptions properly.

```php
try {
    $user = $this->model->find($id);
    if (!$user) {
        throw new \Exception('User not found');
    }
} catch (\Exception $e) {
    log_message('error', $e->getMessage());
}
```

---

## **7. Debugging Queries with Query Builder**  
To check database queries, use:  
```php
$query = $this->db->getLastQuery();
log_message('debug', 'Executed Query: ' . $query);
```

or  
```php
echo $this->db->getLastQuery();
```
✅ This helps in **debugging incorrect database queries**.

---

## **8. Debugging Using CodeIgniter’s Debug Toolbar**  
Enable the debug toolbar in **`app/Config/Filters.php`**:
```php
public $globals = [
    'before' => [],
    'after'  => ['toolbar'],
];
```
✅ The toolbar will display **execution time, memory usage, and SQL queries**.

---

## **9. Debugging AJAX Requests**  
When debugging AJAX, errors won’t be visible in the browser. Use:
```php
log_message('error', 'AJAX request failed');
```
Check logs in `writable/logs/`.

---

## **10. Testing Error Pages (404 & 500)**
### **Handle 404 Errors**
Modify **`app/Config/Routes.php`**:
```php
$routes->set404Override(function () {
    echo view('errors/custom_404');
});
```

📁 **Create `app/Views/errors/custom_404.php`**
```html
<h1>404 - Page Not Found</h1>
<p>The page you requested does not exist.</p>
```

### **Handle 500 Errors**
Modify **`app/Config/Exceptions.php`**:
```php
public $sensitiveDataInTrace = false;
```

---

## **Conclusion**  
✅ **Enable error reporting** for debugging.  
✅ **Use logs** to track errors in production.  
✅ **Use debugging tools** like `log_message()` and `var_dump()`.  
✅ **Debug database queries** using `getLastQuery()`.  
✅ **Enable the Debug Toolbar** for insights.  

🚀 Now, you can effectively debug and log issues in CodeIgniter!