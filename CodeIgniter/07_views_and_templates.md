# **Creating Dynamic Views and Using Templates in CodeIgniter**  

In CodeIgniter, **views** handle the presentation layer, allowing you to separate business logic from UI. You can use **dynamic views and templates** to improve maintainability and reusability.

---

## **1. Understanding Views in CodeIgniter**  

A **view** is a simple PHP file located in `app/Views/`. It contains **HTML, CSS, and embedded PHP** to display data dynamically.

### **Example: Creating a Simple View**
📁 `app/Views/welcome_message.php`
```php
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Welcome</title>
</head>
<body>
    <h1>Welcome to CodeIgniter</h1>
    <p><?= esc($message) ?></p>
</body>
</html>
```
📌 **Using the View in a Controller**  
📁 `app/Controllers/Home.php`
```php
<?php

namespace App\Controllers;

class Home extends BaseController
{
    public function index()
    {
        $data['message'] = "This is a dynamic message!";
        return view('welcome_message', $data);
    }
}
```
🔗 Visit `http://localhost/my_project/home`  
✅ Output: Displays the welcome page with the dynamic message.

---

## **2. Passing Dynamic Data to Views**  
You can pass data from a controller to a view as an **associative array**.

📁 `app/Controllers/User.php`
```php
public function profile()
{
    $data = [
        'username' => 'Sandeep',
        'email' => 'sandeep@example.com'
    ];
    return view('user_profile', $data);
}
```
📁 `app/Views/user_profile.php`
```php
<h1>Welcome, <?= esc($username) ?></h1>
<p>Email: <?= esc($email) ?></p>
```
🔗 Visiting `http://localhost/my_project/user/profile` will show:  
✅ **"Welcome, Sandeep"**  
✅ **"Email: sandeep@example.com"**

---

## **3. Creating a Master Template for Reusability**  
Instead of repeating HTML headers and footers in multiple files, use a **template system**.

### **Step 1: Create a Layout Template**  
📁 `app/Views/layouts/main.php`
```php
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title><?= esc($title) ?></title>
</head>
<body>
    <header>
        <h1>My Website</h1>
    </header>

    <main>
        <?= $this->renderSection('content') ?>
    </main>

    <footer>
        <p>&copy; 2025 My Website</p>
    </footer>
</body>
</html>
```

### **Step 2: Create a Page Extending the Template**  
📁 `app/Views/pages/home.php`
```php
<?= $this->extend('layouts/main') ?>

<?= $this->section('content') ?>
    <h2>Welcome to My Website</h2>
    <p>This is the homepage.</p>
<?= $this->endSection() ?>
```

### **Step 3: Load the View in a Controller**  
📁 `app/Controllers/Page.php`
```php
<?php

namespace App\Controllers;

class Page extends BaseController
{
    public function home()
    {
        $data['title'] = "Home";
        return view('pages/home', $data);
    }
}
```
🔗 Visit `http://localhost/my_project/page/home`  
✅ **Header, Footer, and Content appear dynamically!**

---

## **4. Using View Fragments (Headers and Footers)**  
Another way to reuse content is by separating the **header and footer**.

📁 `app/Views/templates/header.php`
```php
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title><?= esc($title) ?></title>
</head>
<body>
<header>
    <h1>My Website</h1>
</header>
```
📁 `app/Views/templates/footer.php`
```php
<footer>
    <p>&copy; 2025 My Website</p>
</footer>
</body>
</html>
```
📁 `app/Views/about.php`
```php
<?php echo view('templates/header', ['title' => 'About Us']); ?>
<h2>About Us</h2>
<p>We provide the best services.</p>
<?php echo view('templates/footer'); ?>
```
🔗 Visit `http://localhost/my_project/page/about`  
✅ **Header and Footer appear dynamically!**

---

## **5. Using Components in Views**  
For reusable elements like a **navigation menu**, create a component.

📁 `app/Views/templates/navbar.php`
```php
<nav>
    <a href="/">Home</a> | 
    <a href="/about">About</a> | 
    <a href="/contact">Contact</a>
</nav>
```
📁 `app/Views/pages/contact.php`
```php
<?php echo view('templates/header', ['title' => 'Contact Us']); ?>
<?php echo view('templates/navbar'); ?>
<h2>Contact Us</h2>
<p>Email: contact@example.com</p>
<?php echo view('templates/footer'); ?>
```
🔗 ✅ Navbar appears dynamically on every page!

---

## **6. Using Sections and Layouts in CodeIgniter**  
Using **extend()**, **section()**, and **endSection()** allows for flexible templating.

Example:
```php
<?= $this->extend('layouts/main') ?>

<?= $this->section('content') ?>
    <h2>Dashboard</h2>
    <p>Welcome to the admin dashboard.</p>
<?= $this->endSection() ?>
```
✅ **Easier to maintain and scale!**

---

## **7. Using View Cells for Reusable Components**  
View Cells allow you to inject components dynamically.

📁 `app/Cells/RecentPosts.php`
```php
<?php namespace App\Cells;

use CodeIgniter\View\Cells\Cell;

class RecentPosts extends Cell
{
    public function render()
    {
        return "<ul><li>Post 1</li><li>Post 2</li></ul>";
    }
}
```
📁 `app/Views/pages/blog.php`
```php
<h2>Latest Blog Posts</h2>
<?= view_cell('App\Cells\RecentPosts') ?>
```
✅ View Cells enable **reusable components** dynamically.

---

## **Conclusion**  
✔ **Views separate UI from logic**  
✔ **Pass data dynamically from controllers**  
✔ **Templates improve reusability**  
✔ **Extend layouts for structured design**  
✔ **View fragments (header/footer) prevent repetition**  
✔ **View Cells enhance modularity**  

🚀 **Next:** Would you like a tutorial on **CodeIgniter Models & Database Handling**?