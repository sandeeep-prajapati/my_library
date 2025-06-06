# **How to Implement Multilingual Support in CodeIgniter Applications?**  

Implementing multilingual support in CodeIgniter allows your application to support multiple languages dynamically. This guide covers **loading language files, switching languages, and setting up translations** in CodeIgniter.

---

## **1. Enable Language Support in CodeIgniter**  
CodeIgniter has a built-in **Language Class** that loads language files from `app/Language/`.

📁 **Folder Structure:**  
```
app/
 ├── Language/
 │   ├── english/
 │   │   ├── messages.php
 │   ├── hindi/
 │   │   ├── messages.php
```

✅ **Each language has a separate folder containing translation files.**

---

## **2. Create Language Files**  
Language files are PHP files containing an array of key-value pairs.

### **Example: English Language File**  
📁 **`app/Language/english/messages.php`**
```php
<?php
return [
    'welcome_message' => 'Welcome to our website!',
    'login_success'   => 'You have successfully logged in.',
    'logout_message'  => 'You have logged out successfully.',
];
```

### **Example: Hindi Language File**  
📁 **`app/Language/hindi/messages.php`**
```php
<?php
return [
    'welcome_message' => 'हमारी वेबसाइट में आपका स्वागत है!',
    'login_success'   => 'आप सफलतापूर्वक लॉग इन हो गए हैं।',
    'logout_message'  => 'आप सफलतापूर्वक लॉग आउट हो गए हैं।',
];
```

---

## **3. Load Language Files in Controllers**  
Use the `Language` class to load translations dynamically.

📁 **Example Controller: `app/Controllers/Home.php`**
```php
<?php

namespace App\Controllers;
use CodeIgniter\Controller;
use CodeIgniter\I18n\Time;

class Home extends Controller
{
    public function index()
    {
        // Load language file
        $session = session();
        $language = $session->get('lang') ?? 'english';
        service('language')->setLocale($language);

        return view('welcome_message');
    }
}
```
✅ This ensures the selected language is used across the application.

---

## **4. Display Translations in Views**  
Use the `lang()` helper function to display translated text.

📁 **Example View: `app/Views/welcome_message.php`**
```php
<h1><?= lang('messages.welcome_message'); ?></h1>
<p><?= lang('messages.login_success'); ?></p>
```
✅ The text will change based on the selected language.

---

## **5. Create a Language Switcher**  
Allow users to switch between languages using a dropdown.

📁 **Example Controller: `app/Controllers/Language.php`**
```php
<?php

namespace App\Controllers;
use CodeIgniter\Controller;

class Language extends Controller
{
    public function switch($lang)
    {
        $session = session();
        $session->set('lang', $lang);

        return redirect()->to(previous_url()); // Redirect back
    }
}
```

📁 **Add Routes in `app/Config/Routes.php`**
```php
$routes->get('language/switch/(:segment)', 'Language::switch/$1');
```

📁 **Example Language Switcher in `app/Views/navbar.php`**
```php
<a href="<?= site_url('language/switch/english'); ?>">English</a> |
<a href="<?= site_url('language/switch/hindi'); ?>">हिन्दी</a>
```
✅ Clicking on these links will switch the language.

---

## **6. Auto-Detect Browser Language (Optional)**  
Set the language based on the user’s browser preference.

📁 **Modify `app/Controllers/Home.php`**
```php
public function index()
{
    $session = session();
    if (!$session->has('lang')) {
        $language = substr($_SERVER['HTTP_ACCEPT_LANGUAGE'], 0, 2) == 'hi' ? 'hindi' : 'english';
        $session->set('lang', $language);
    }
    service('language')->setLocale($session->get('lang'));

    return view('welcome_message');
}
```
✅ Automatically sets the language based on the browser.

---

## **7. Store Language Preference in Cookies (Optional)**  
Instead of sessions, store the user’s language choice in a cookie.

📁 **Modify `app/Controllers/Language.php`**
```php
public function switch($lang)
{
    $cookie = [
        'name'   => 'user_lang',
        'value'  => $lang,
        'expire' => 365 * 24 * 60 * 60, // 1 year
        'secure' => false
    ];
    set_cookie($cookie);

    return redirect()->to(previous_url());
}
```
📁 **Modify `app/Controllers/Home.php`**
```php
$language = get_cookie('user_lang') ?? 'english';
service('language')->setLocale($language);
```
✅ This ensures the language preference is **remembered across sessions**.

---

## **Conclusion**  
✅ **Store translations** in `app/Language/` directories.  
✅ **Use `lang()` helper** to display translations dynamically.  
✅ **Switch languages** using a controller and session.  
✅ **Auto-detect browser language** for a better user experience.  
✅ **Store language settings** in cookies for persistent preferences.  

🚀 Now, your CodeIgniter app supports multiple languages!