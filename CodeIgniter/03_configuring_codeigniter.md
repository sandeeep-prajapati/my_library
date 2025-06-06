# **Configuring Base URL, Database Settings, and Autoload Components in CodeIgniter**  

Before developing a CodeIgniter application, it’s essential to configure the base URL, database settings, and autoload components. This ensures smooth functionality and reduces the need for repetitive configuration.  

---

## **1. Configuring the Base URL**  

The **base URL** defines the root address of your CodeIgniter project.  

### **Steps to Set the Base URL**  

1. **Open the `app/Config/App.php` file.**  
2. Locate the following line:  

   ```php
   public $baseURL = 'http://localhost:8080/';
   ```
3. **Update it according to your project’s URL.**  
   - For **XAMPP**:  
     ```php
     public $baseURL = 'http://localhost/my_project/';
     ```
   - For **Live Server**:  
     ```php
     public $baseURL = 'https://mywebsite.com/';
     ```
4. **Save the file** and restart the server if needed.  

---

## **2. Configuring Database Settings**  

If your project interacts with a database, you must configure database credentials.  

### **Steps to Set Up the Database**  

1. **Open the `app/Config/Database.php` file.**  
2. Find the **default database configuration**:  

   ```php
   public $default = [
       'DSN'      => '',
       'hostname' => 'localhost',
       'username' => 'root',
       'password' => '',
       'database' => 'my_database',
       'DBDriver' => 'MySQLi',
       'DBPrefix' => '',
       'pConnect' => false,
       'DBDebug'  => true,
       'cacheOn'  => false,
       'charset'  => 'utf8',
       'DBCollat' => 'utf8_general_ci',
       'swapPre'  => '',
       'encrypt'  => false,
       'compress' => false,
       'strictOn' => false,
       'failover' => [],
       'port'     => 3306,
   ];
   ```
3. **Modify the credentials** based on your database:  
   - **For XAMPP/WAMP:**  
     ```php
     'hostname' => 'localhost',
     'username' => 'root',
     'password' => '',
     'database' => 'my_database',
     ```
   - **For Remote Database:**  
     ```php
     'hostname' => 'your_host',
     'username' => 'your_user',
     'password' => 'your_password',
     'database' => 'your_database',
     ```
4. **Save the file** and restart the application.  

### **Testing Database Connection**  
Run the following command in the terminal:  

```sh
php spark db:table
```
If everything is set correctly, it will display database tables.  

---

## **3. Configuring Autoload Components**  

Autoloading allows you to automatically load helpers, libraries, and models instead of manually loading them in each controller.  

### **Steps to Autoload Components**  

1. **Open `app/Config/Autoload.php`**  
2. Locate the **helpers array** and add frequently used helpers:  

   ```php
   public $helpers = ['url', 'form', 'html'];
   ```
   - `url`: Helps with URL generation (`base_url()`, `site_url()`).  
   - `form`: Simplifies form creation (`form_open()`, `form_input()`).  
   - `html`: Helps with HTML elements (`img()`, `link_tag()`).  

3. Locate the **libraries array** and add libraries like session and validation:  

   ```php
   public $libraries = ['session', 'database'];
   ```
   - `session`: Manages user sessions.  
   - `database`: Ensures database access in all controllers.  

4. Locate the **models array** and add models:  

   ```php
   public $models = ['StudentModel'];
   ```
   This automatically loads the `StudentModel` in all controllers.  

5. **Save the file** and restart the server.  

---

## **4. Additional Configurations**  

### **Configuring `.env` File for Environment Variables**  
1. Rename `.env.example` to `.env`.  
2. Uncomment and set the `CI_ENVIRONMENT` to **development** (for debugging) or **production** (for live applications):  

   ```
   CI_ENVIRONMENT = development
   ```

3. Configure database settings in `.env` instead of `Database.php`:  

   ```
   database.default.hostname = localhost
   database.default.database = my_database
   database.default.username = root
   database.default.password =
   database.default.DBDriver = MySQLi
   ```

---

## **5. Testing the Configuration**  

### **Check Base URL**  
Visit: `http://localhost/my_project/`  

### **Check Database Connection**  
Run a test query in a controller:  

```php
public function testDB()
{
    $db = \Config\Database::connect();
    $query = $db->query('SELECT * FROM users');
    print_r($query->getResult());
}
```
Visit: `http://localhost/my_project/testDB`  

### **Check Autoloaded Helpers**  
Use `base_url()` in a view:  

```php
<a href="<?= base_url('home'); ?>">Home</a>
```

---

## **Conclusion**  
✔ **Base URL** ensures correct routing.  
✔ **Database configuration** enables interaction with MySQL.  
✔ **Autoloading** simplifies helper and library usage.  
