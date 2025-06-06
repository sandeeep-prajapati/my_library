# **Composer: PHP Dependency Management**  

Composer is a dependency manager for PHP, allowing developers to manage libraries and packages efficiently. It simplifies package installation, updates, and autoloading in PHP projects.  

---

## **1. What is Composer?**  
Composer is a tool that:  
✅ Installs and updates third-party PHP packages.  
✅ Resolves dependencies automatically.  
✅ Uses **Packagist** (the main repository for PHP packages).  
✅ Provides **autoloading** to avoid manual `require` statements.  

🔹 **Installation:** Download Composer from [getcomposer.org](https://getcomposer.org).  

```bash
php -r "copy('https://getcomposer.org/installer', 'composer-setup.php');"
php composer-setup.php
php -r "unlink('composer-setup.php');"
```

Verify installation:  
```bash
composer --version
```

---

## **2. `composer.json` – Defining Dependencies**  
The `composer.json` file stores package details and project metadata.  

### **🚀 Example: A Simple `composer.json` File**  
```json
{
  "name": "myproject/app",
  "description": "A sample PHP project",
  "require": {
    "monolog/monolog": "^3.0"
  },
  "autoload": {
    "psr-4": {
      "App\\": "src/"
    }
  }
}
```
🔹 **Key Sections:**  
- `"require"`: Lists dependencies (e.g., `monolog/monolog`).  
- `"autoload"`: Defines PSR-4 autoloading for classes.  

---

## **3. Installing Dependencies**  
After defining dependencies, install them with:  
```bash
composer install
```
This creates a `vendor/` directory and generates a `composer.lock` file.  

To add new packages:  
```bash
composer require guzzlehttp/guzzle
```
This automatically updates `composer.json` and downloads the package.  

To update all dependencies:  
```bash
composer update
```

---

## **4. Autoloading Classes**  
Composer provides **automatic class loading**, eliminating the need for manual `require` statements.  

### **🚀 Example: Using Autoloading**
1️⃣ Define a class in `src/Greeting.php`:  
```php
namespace App;

class Greeting {
    public function sayHello() {
        return "Hello, Composer!";
    }
}
```
  
2️⃣ Include the autoloader in `index.php`:  
```php
require 'vendor/autoload.php';

use App\Greeting;

$greet = new Greeting();
echo $greet->sayHello();
```
Now, running `php index.php` outputs:  
```
Hello, Composer!
```

🔹 **Why Use Composer's Autoload?**  
✔️ Automatically loads classes based on namespaces.  
✔️ Supports **PSR-4 autoloading** for structured projects.  
✔️ Reduces unnecessary `require` statements.  

---

## **5. Useful Composer Commands**  
🔹 **Check outdated packages:**  
```bash
composer outdated
```
🔹 **Remove a package:**  
```bash
composer remove monolog/monolog
```
🔹 **Dump autoload (when adding new classes manually):**  
```bash
composer dump-autoload
```

---

## **Conclusion**  
Composer is essential for PHP development, making dependency management and autoloading seamless. 🚀  

Would you like a **real-world project setup using Composer?** 💡