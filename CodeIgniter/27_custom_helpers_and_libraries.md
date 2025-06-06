# **How to Use Composer to Manage Dependencies in CodeIgniter?**  

Composer is a dependency manager for PHP that helps manage libraries and packages efficiently. CodeIgniter supports Composer for integrating third-party libraries into your project.

---

## **1. Install Composer (If Not Installed)**
First, ensure you have **Composer** installed on your system.

🔹 **Check if Composer is installed:**
```sh
composer -V
```
🔹 **If not installed, download and install it from:**  
👉 [https://getcomposer.org/download/](https://getcomposer.org/download/)

---

## **2. Enable Composer in CodeIgniter**  
CodeIgniter 4 comes with built-in **Composer support**, but you need to enable it.

📁 **Edit `app/Config/Constants.php`**
```php
defined('COMPOSER_PATH') || define('COMPOSER_PATH', ROOTPATH . 'vendor/autoload.php');
```
✅ This tells CodeIgniter to load dependencies from `vendor/autoload.php`.

---

## **3. Install Dependencies Using Composer**  
To install a package, use the following command:

🔹 **Example: Install `phpmailer/phpmailer` for sending emails**  
```sh
composer require phpmailer/phpmailer
```
This will download the package to the `vendor/` directory.

---

## **4. Load Composer Packages in CodeIgniter**  
Once installed, load the package in your CodeIgniter controller.

📁 **Example: Using PHPMailer in `app/Controllers/EmailController.php`**
```php
<?php
namespace App\Controllers;
use PHPMailer\PHPMailer\PHPMailer;
use PHPMailer\PHPMailer\Exception;

class EmailController extends BaseController
{
    public function sendEmail()
    {
        $mail = new PHPMailer(true);
        
        try {
            $mail->isSMTP();
            $mail->Host = 'smtp.example.com';
            $mail->SMTPAuth = true;
            $mail->Username = 'your_email@example.com';
            $mail->Password = 'your_password';
            $mail->SMTPSecure = 'tls';
            $mail->Port = 587;

            $mail->setFrom('your_email@example.com', 'Your Name');
            $mail->addAddress('recipient@example.com');

            $mail->Subject = 'Test Email';
            $mail->Body    = 'This is a test email using PHPMailer in CodeIgniter.';

            $mail->send();
            echo "Email sent successfully!";
        } catch (Exception $e) {
            echo "Mailer Error: " . $mail->ErrorInfo;
        }
    }
}
```
✅ Now, PHPMailer works in your CodeIgniter application.

---

## **5. Remove a Package (If Needed)**  
To remove an installed package, use:
```sh
composer remove phpmailer/phpmailer
```
This deletes the package from `vendor/` and `composer.json`.

---

## **6. Update Dependencies**  
To update all dependencies:
```sh
composer update
```
To update a specific package:
```sh
composer update phpmailer/phpmailer
```

---

## **7. Autoload Custom Libraries with Composer**  
If you have a custom PHP library in `app/Libraries`, tell Composer to autoload it.

📁 **Modify `composer.json`**
```json
"autoload": {
    "psr-4": {
        "App\\Libraries\\": "app/Libraries/"
    }
}
```
Run **`composer dump-autoload`** to update autoloading.

✅ Now, you can use your custom libraries like this:
```php
use App\Libraries\MyCustomLibrary;
$library = new MyCustomLibrary();
```

---

## **Conclusion**  
✅ **Install Composer** and enable it in CodeIgniter.  
✅ **Use `composer require`** to install packages.  
✅ **Load dependencies using `vendor/autoload.php`**.  
✅ **Manage autoloading for custom libraries**.  
✅ **Update and remove packages easily**.  

🚀 **Now your CodeIgniter project is ready to use Composer efficiently!**