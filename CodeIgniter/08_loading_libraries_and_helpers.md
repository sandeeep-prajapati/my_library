# **Using Built-in Libraries and Helpers in CodeIgniter**  

CodeIgniter provides **built-in libraries** and **helpers** to simplify common tasks such as form validation, session management, file handling, and more.  

---

## **1. Understanding Libraries and Helpers**  
🔹 **Libraries** are **object-oriented classes** that provide reusable functionality.  
🔹 **Helpers** are **collections of procedural functions** for common tasks.  

📍 **Example:**  
- The **Session Library** provides session management.  
- The **URL Helper** provides functions for working with URLs.  

---

## **2. Loading and Using Libraries**  

📌 **Automatic Loading (Recommended for frequently used libraries)**  
Edit **`app/Config/Autoload.php`** to load libraries automatically:  
```php
public $libraries = ['session', 'email'];
```

📌 **Manual Loading (For specific controllers/methods only)**  
```php
$this->load->library('session');
```

---

### **3. Commonly Used Built-in Libraries**  

### **A. Session Library** (Managing user sessions)  
📁 `app/Controllers/User.php`  
```php
<?php
namespace App\Controllers;

class User extends BaseController
{
    public function setSession()
    {
        session()->set([
            'username' => 'Sandeep',
            'logged_in' => true
        ]);
        return "Session Set!";
    }

    public function getSession()
    {
        echo session()->get('username');  // Output: Sandeep
    }

    public function destroySession()
    {
        session()->destroy();
        return "Session Destroyed!";
    }
}
```
🔗 **Access:**  
- `http://localhost/my_project/user/setSession`  
- `http://localhost/my_project/user/getSession`  
- `http://localhost/my_project/user/destroySession`  

✅ **Session data stored and retrieved successfully!**  

---

### **B. Form Validation Library**  
📁 `app/Controllers/Form.php`  
```php
<?php
namespace App\Controllers;

use CodeIgniter\Controller;

class Form extends Controller
{
    public function validateInput()
    {
        $validation = \Config\Services::validation();

        $rules = [
            'name'  => 'required|min_length[3]',
            'email' => 'required|valid_email'
        ];

        if (!$this->validate($rules)) {
            return view('form', ['validation' => $this->validator]);
        } else {
            return "Form Submitted Successfully!";
        }
    }
}
```
📁 `app/Views/form.php`  
```php
<form action="/form/validateInput" method="post">
    <input type="text" name="name" placeholder="Name">
    <input type="email" name="email" placeholder="Email">
    <button type="submit">Submit</button>
</form>

<?php if (isset($validation)) : ?>
    <?= $validation->listErrors() ?>
<?php endif; ?>
```
🔗 **Visit:** `http://localhost/my_project/form`  

✅ **Displays errors if validation fails!**  

---

### **C. Email Library (Sending Emails)**  
📁 `app/Controllers/Mail.php`  
```php
<?php
namespace App\Controllers;

use CodeIgniter\Controller;

class Mail extends Controller
{
    public function sendEmail()
    {
        $email = \Config\Services::email();

        $email->setFrom('your-email@example.com', 'Your Name');
        $email->setTo('receiver@example.com');
        $email->setSubject('Test Email');
        $email->setMessage('This is a test email from CodeIgniter.');

        if ($email->send()) {
            return "Email sent successfully!";
        } else {
            return $email->printDebugger(['headers']);
        }
    }
}
```
✅ **Configurable via `app/Config/Email.php`**  
✅ **SMTP or Mail protocol support!**  

---

## **4. Loading and Using Helpers**  
📌 **Automatic Loading (Recommended for commonly used helpers)**  
Edit `app/Config/Autoload.php`:  
```php
public $helpers = ['url', 'form'];
```

📌 **Manual Loading (For specific controllers/methods only)**  
```php
helper('url');
helper(['form', 'text']);
```

---

### **5. Commonly Used Built-in Helpers**  

### **A. URL Helper (Generating URLs, Redirecting)**  
🔹 **Creating Links Dynamically**  
```php
echo anchor('user/profile', 'Go to Profile'); 
// Output: <a href="http://localhost/my_project/user/profile">Go to Profile</a>
```
🔹 **Redirecting Users**  
```php
return redirect()->to('/dashboard');
```

---

### **B. Form Helper (Simplifying Forms)**  
```php
echo form_open('/form/submit');
echo form_input('username', 'Sandeep');
echo form_submit('submit', 'Submit');
echo form_close();
```
✅ **Generates properly formatted form elements.**  

---

### **C. Text Helper (Generating Random Strings)**  
```php
helper('text');
echo random_string('alnum', 8);  // Example Output: 4G8kd9W2
```
✅ **Useful for unique IDs or verification codes.**  

---

### **D. Security Helper (Preventing XSS Attacks)**  
```php
helper('security');
$clean_input = xss_clean($user_input);
```
✅ **Protects against XSS (Cross-Site Scripting).**  

---

## **Conclusion**  
✔ **Libraries** (Object-Oriented) are loaded via `$this->load->library('library_name')`  
✔ **Helpers** (Procedural Functions) are loaded via `helper('helper_name')`  
✔ **Sessions, Form Validation, Emails, and Security Helpers are crucial**  
✔ **Helpers like URL, Form, and Text simplify development**  

🚀 **Next:** Would you like a tutorial on **working with databases using CodeIgniter Models?**