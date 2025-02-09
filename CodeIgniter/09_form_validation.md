# **Implementing Form Validation in CodeIgniter**  

Form validation is crucial for ensuring data integrity and preventing security vulnerabilities. CodeIgniter provides a **built-in validation library** that simplifies the process.  

---

## **1. Loading the Form Validation Library**  
CodeIgniter **automatically** loads the validation library when you use `$this->validate()`, but you can also load it manually:  

```php
$validation = \Config\Services::validation();
```

Alternatively, **autoload it** in `app/Config/Autoload.php`:  

```php
public $libraries = ['validation'];
```

---

## **2. Setting Validation Rules**  

### **A. Defining Rules in the Controller**  
📁 `app/Controllers/Form.php`  

```php
<?php
namespace App\Controllers;

use CodeIgniter\Controller;

class Form extends Controller
{
    public function index()
    {
        return view('form_view');
    }

    public function submit()
    {
        $validation = \Config\Services::validation();

        $rules = [
            'name'  => 'required|min_length[3]|max_length[50]',
            'email' => 'required|valid_email',
            'age'   => 'required|integer|greater_than[18]'
        ];

        if (!$this->validate($rules)) {
            return view('form_view', ['validation' => $this->validator]);
        } else {
            return "Form Submitted Successfully!";
        }
    }
}
```

✅ **Explanation of Rules:**  
- `required` → Field cannot be empty  
- `min_length[3]` → Minimum 3 characters  
- `max_length[50]` → Maximum 50 characters  
- `valid_email` → Must be a valid email  
- `integer` → Must be a number  
- `greater_than[18]` → Must be greater than 18  

---

## **3. Creating the View with Error Messages**  
📁 `app/Views/form_view.php`  

```php
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Form Validation</title>
</head>
<body>
    <h2>Form Validation in CodeIgniter</h2>

    <?php if (isset($validation)) : ?>
        <div style="color: red;">
            <?= $validation->listErrors(); ?>
        </div>
    <?php endif; ?>

    <form action="<?= site_url('form/submit') ?>" method="post">
        <label>Name:</label>
        <input type="text" name="name"><br><br>

        <label>Email:</label>
        <input type="email" name="email"><br><br>

        <label>Age:</label>
        <input type="number" name="age"><br><br>

        <button type="submit">Submit</button>
    </form>
</body>
</html>
```

---

## **4. Custom Error Messages**  
You can define **custom error messages** in `app/Controllers/Form.php`:  

```php
$rules = [
    'name' => [
        'label'  => 'Full Name',
        'rules'  => 'required|min_length[3]',
        'errors' => [
            'required'    => '{field} is required.',
            'min_length'  => '{field} must have at least {param} characters.'
        ]
    ],
    'email' => [
        'rules'  => 'required|valid_email',
        'errors' => [
            'valid_email' => 'Please provide a valid email address.'
        ]
    ]
];
```

✅ **Now, errors will display more readable messages!**  

---

## **5. Validating Checkbox, Radio, and Dropdown Fields**  

### **A. Checkbox Validation**  
```php
'terms' => 'required'
```

📁 **HTML View:**  
```php
<input type="checkbox" name="terms" value="1"> I agree to the terms and conditions
```

### **B. Radio Button Validation**  
```php
'gender' => 'required'
```

📁 **HTML View:**  
```php
<input type="radio" name="gender" value="male"> Male
<input type="radio" name="gender" value="female"> Female
```

### **C. Dropdown Validation**  
```php
'country' => 'required'
```

📁 **HTML View:**  
```php
<select name="country">
    <option value="">Select Country</option>
    <option value="India">India</option>
    <option value="USA">USA</option>
</select>
```

---

## **6. Redirecting with Flash Messages (Success or Error)**  
Instead of returning plain text, we can **redirect the user with success/error messages**:  

📁 **Modify `submit()` in `Form.php`**  
```php
public function submit()
{
    $session = session();
    $validation = \Config\Services::validation();

    $rules = [
        'name'  => 'required|min_length[3]',
        'email' => 'required|valid_email'
    ];

    if (!$this->validate($rules)) {
        $session->setFlashdata('errors', $this->validator->listErrors());
        return redirect()->to('/form')->withInput();
    }

    $session->setFlashdata('success', 'Form submitted successfully!');
    return redirect()->to('/form');
}
```

📁 **Update `form_view.php` to show flash messages**  
```php
<?php if (session()->getFlashdata('errors')) : ?>
    <div style="color: red;">
        <?= session()->getFlashdata('errors') ?>
    </div>
<?php endif; ?>

<?php if (session()->getFlashdata('success')) : ?>
    <div style="color: green;">
        <?= session()->getFlashdata('success') ?>
    </div>
<?php endif; ?>
```

✅ **This ensures users see validation messages after redirection!**  

---

## **7. Client-Side and Server-Side Validation**  
It’s **recommended** to combine server-side validation with **JavaScript validation**:  

📁 **Add JavaScript validation to `form_view.php`**  
```html
<script>
document.querySelector("form").addEventListener("submit", function(event) {
    let name = document.querySelector("input[name='name']").value;
    let email = document.querySelector("input[name='email']").value;

    if (name.length < 3) {
        alert("Name must be at least 3 characters long.");
        event.preventDefault();
    }

    let emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailPattern.test(email)) {
        alert("Please enter a valid email.");
        event.preventDefault();
    }
});
</script>
```
✅ **Prevents unnecessary form submissions!**  

---

## **8. Database Validation (Checking Unique Values)**  
To **ensure a value is unique in the database**, modify validation rules:  

```php
'email' => 'required|valid_email|is_unique[users.email]'
```

📌 This checks if the email **already exists** in the `users` table.  

✅ **Prevents duplicate entries in the database!**  

---

## **Final Thoughts**  
✔ **CodeIgniter’s Form Validation Library** makes validation easy.  
✔ **Use server-side validation** to ensure security.  
✔ **Enhance UX with JavaScript validation**.  
✔ **Store errors in session flashdata** for better user experience.  
✔ **Use `is_unique` rule** for database validation.  

🚀 **Next Step:** Would you like to integrate form validation with database storage? 😊