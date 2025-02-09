# **Best Security Practices in CodeIgniter: Preventing SQL Injection, XSS, and CSRF Attacks**  

Security is crucial in any web application. CodeIgniter provides built-in mechanisms to prevent **SQL Injection, Cross-Site Scripting (XSS), and Cross-Site Request Forgery (CSRF)** attacks. Let’s explore how to secure your CodeIgniter application effectively.  

---

## **1. Preventing SQL Injection**  

SQL Injection occurs when malicious SQL queries are injected into your database through input fields.  

### ✅ **Use Query Builder (Active Record) Instead of Raw Queries**  

Avoid:  
```php
$this->db->query("SELECT * FROM users WHERE email = '$email' AND password = '$password'");
```
✅ **Use parameterized queries instead:**  
```php
$this->db->where('email', $email);
$this->db->where('password', $password);
$query = $this->db->get('users');
```

✅ **Or Use `query()` with Binding**  
```php
$query = $this->db->query("SELECT * FROM users WHERE email = ? AND password = ?", [$email, $password]);
```
🔒 **Why?** It prevents attackers from injecting SQL commands.

---

## **2. Preventing XSS (Cross-Site Scripting)**  

XSS attacks happen when an attacker injects malicious scripts into a webpage viewed by users.  

### ✅ **Sanitize User Input Using `xss_clean()`**  
```php
$this->load->helper('security');
$clean_input = $this->security->xss_clean($this->input->post('comment'));
```
🔒 **Why?** It removes harmful scripts like `<script>alert('Hacked!')</script>`.

### ✅ **Use Escaped Output in Views**  
Instead of:  
```php
echo $user_comment;
```
Use:  
```php
echo esc($user_comment); // Escapes HTML special characters
```
🔒 **Why?** Prevents stored XSS in databases.

---

## **3. Preventing CSRF (Cross-Site Request Forgery)**  

CSRF attacks trick authenticated users into making unwanted requests.  

### ✅ **Enable CSRF Protection in `Config/Filters.php`**  
In **CodeIgniter 4**, CSRF protection is managed by filters.  

📁 `app/Config/Filters.php`  
```php
public $globals = [
    'before' => [
        'csrf' // Enable CSRF globally
    ]
];
```

### ✅ **Use CSRF Token in Forms**  
In views:  
```php
<form method="post" action="<?= base_url('submit-form') ?>">
    <?= csrf_field() ?> <!-- Adds CSRF token -->
    <input type="text" name="name">
    <button type="submit">Submit</button>
</form>
```
🔒 **Why?** Ensures only forms with a valid CSRF token are submitted.

---

## **4. Other Security Best Practices**  

### ✅ **Use Password Hashing Instead of Plaintext Storage**  
```php
$password = password_hash($user_password, PASSWORD_BCRYPT);
```
✅ To verify:  
```php
if (password_verify($entered_password, $hashed_password)) {
    echo "Login successful";
}
```
🔒 **Why?** Protects against password leaks.

### ✅ **Restrict File Uploads to Safe Types**  
📁 `app/Config/Mimes.php` (Define allowed file types)  
```php
'jpg'  => ['image/jpeg', 'image/pjpeg'],
'png'  => ['image/png'],
'gif'  => ['image/gif'],
```
✅ Use `mime_content_type()` to validate files.

---

## **Final Thoughts**  

🔹 **SQL Injection** → Use Query Builder or Prepared Statements  
🔹 **XSS** → Sanitize input with `xss_clean()` and escape output  
🔹 **CSRF** → Enable CSRF protection and use CSRF tokens  
🔹 **Passwords** → Always hash using `password_hash()`  

💡 **Security is an ongoing process!** Keep updating your CodeIgniter app to the latest version for security patches. 🚀