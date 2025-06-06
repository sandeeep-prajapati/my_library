## **How to Manage Sessions and Cookies in CodeIgniter?**  

Managing **sessions and cookies** in CodeIgniter is essential for user authentication, tracking user activity, and storing temporary data. CodeIgniter provides built-in support for handling both efficiently.  

---

## **1. Configuring Session in CodeIgniter**  

📁 **Configuration File:** `app/Config/Session.php`  

```php
public $driver = 'CodeIgniter\Session\Handlers\FileHandler'; // Default driver
public $cookieName = 'ci_session';
public $expiration = 7200;  // Session lasts 2 hours
public $savePath = WRITEPATH . 'session';  // Path where session files are stored
public $matchIP = false;  
public $timeToUpdate = 300;  
public $regenerateDestroy = false;
```
✅ Adjust session expiration and save path as per requirements.  

---

## **2. Loading the Session Library**  

CodeIgniter automatically loads sessions if configured. However, if needed, manually load it in your controller:  

```php
$session = session();
```
✅ Now you can set, get, and remove session data.  

---

## **3. Setting Session Data**  

```php
$session->set([
    'user_id'  => 1,
    'username' => 'JohnDoe',
    'email'    => 'john@example.com',
    'logged_in'=> true
]);
```
✅ Stores user details in the session.  

---

## **4. Retrieving Session Data**  

```php
$username = $session->get('username');
echo "Logged in user: " . $username;
```
✅ Fetches the value of `username`.  

### **Fetching All Session Data**  

```php
print_r($session->get());
```
✅ Displays all session variables.  

---

## **5. Checking if Session Key Exists**  

```php
if ($session->has('user_id')) {
    echo "User is logged in";
}
```
✅ Ensures session key exists before accessing it.  

---

## **6. Removing Specific Session Data**  

```php
$session->remove('email');
```
✅ Deletes `email` from session.  

---

## **7. Destroying a Session (Logout)**  

```php
$session->destroy();
```
✅ Logs out the user and clears all session data.  

---

## **8. Using Flash Data (Temporary Data)**  

Flash data is stored **only for the next request**. Useful for messages.  

### **Setting Flash Data**  

```php
$session->setFlashdata('message', 'Login Successful!');
```
✅ Stores a success message for the next page load.  

### **Retrieving Flash Data**  

```php
echo $session->getFlashdata('message');
```
✅ Displays the flash message.  

---

# **Managing Cookies in CodeIgniter**  

Cookies are useful for storing small amounts of persistent data on the user's browser.  

---

## **1. Setting a Cookie**  

```php
$response = service('response');
$response->setCookie('user_token', 'abcd1234', 3600); // 1-hour expiry
```
✅ Stores a cookie named `user_token` with a 1-hour lifespan.  

---

## **2. Retrieving a Cookie**  

```php
$request = service('request');
$token = $request->getCookie('user_token');
echo "User Token: " . $token;
```
✅ Fetches the value of the `user_token` cookie.  

---

## **3. Deleting a Cookie**  

```php
$response->deleteCookie('user_token');
```
✅ Removes the `user_token` cookie from the browser.  

---

## **Conclusion**  

✔ **Sessions** are best for storing temporary user data (e.g., login status).  
✔ **Cookies** are ideal for persistent data that remains after closing the browser.  
✔ **Use Flashdata** for success/error messages that disappear after one request.  

🚀 **Next Step:** Do you want a **login/logout system** using sessions? 😊