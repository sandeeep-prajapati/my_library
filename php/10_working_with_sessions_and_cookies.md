# **Sessions and Cookies in PHP**  

Sessions and cookies are essential for managing user data across multiple pages in PHP. They help track user activity, store preferences, and enable authentication mechanisms.  

---

## **1. What Are Cookies? 🍪**  

A **cookie** is a small file stored on the user's browser. It allows websites to remember user data across different visits.  

### **Setting a Cookie with `setcookie()`**  

```php
<?php
setcookie("username", "Sandeep", time() + (86400 * 7), "/"); // Cookie expires in 7 days
?>
```

🔹 **Parameters:**
- `"username"` → Name of the cookie.
- `"Sandeep"` → Value of the cookie.
- `time() + (86400 * 7)` → Expiration time (7 days).
- `"/"` → Cookie is available across the entire website.

---

### **Retrieving a Cookie**  

```php
<?php
if (isset($_COOKIE["username"])) {
    echo "Welcome back, " . $_COOKIE["username"];
} else {
    echo "Cookie not found!";
}
?>
```

---

### **Deleting a Cookie**  

To delete a cookie, set its expiration time to the past:  

```php
<?php
setcookie("username", "", time() - 3600, "/"); // Cookie expires immediately
?>
```

---

## **2. What Are Sessions? 🔐**  

A **session** stores user data on the server (unlike cookies, which store data in the browser). Sessions are useful for authentication and managing sensitive user information.  

### **Starting a Session with `session_start()`**  

Before using a session, call `session_start()` at the top of your PHP script.  

```php
<?php
session_start();  // Start a session
$_SESSION["user"] = "Sandeep"; // Store session data
echo "Session set successfully!";
?>
```

---

### **Accessing Session Data**  

```php
<?php
session_start();
echo "Hello, " . $_SESSION["user"];
?>
```

---

### **Destroying a Session**  

To log out a user and clear session data:  

```php
<?php
session_start();
session_unset();  // Unset all session variables
session_destroy(); // Destroy the session
echo "Session destroyed!";
?>
```

---

## **3. Cookies vs. Sessions: Key Differences**  

| Feature   | Cookies  | Sessions  |
|-----------|---------|-----------|
| **Storage Location** | Client-side (browser) | Server-side |
| **Data Lifetime** | Until expiration (or deleted) | Until user logs out or session times out |
| **Security** | Less secure (stored in browser) | More secure (stored on server) |
| **Best Use Case** | User preferences (e.g., "Remember Me") | User authentication (e.g., login sessions) |

---

## **4. Best Practices for Managing Sessions & Cookies**  

✅ **Use `secure` and `httponly` flags for cookies:**  
```php
setcookie("authToken", "xyz123", time() + 3600, "/", "", true, true);
```
- `true, true` ensures secure transfer over HTTPS and prevents JavaScript access.

✅ **Regenerate Session ID on Login:**  
```php
session_regenerate_id(true); // Prevents session fixation attacks
```

✅ **Set a session timeout:**  
```php
if (!isset($_SESSION["last_activity"])) {
    $_SESSION["last_activity"] = time();
} elseif (time() - $_SESSION["last_activity"] > 1800) { // 30 mins
    session_unset();
    session_destroy();
}
```

✅ **Avoid storing sensitive data in cookies.**  

Would you like **a practical login system demo** using PHP sessions? 🚀