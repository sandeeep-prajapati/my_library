### **Implementing Form Validation in CodeIgniter**  

Form validation is essential for ensuring data integrity and preventing invalid user input. CodeIgniter provides a **built-in validation library** that makes this process simple and efficient.

---

## **1. Loading the Form Validation Library**  

CodeIgniter **automatically loads** the validation library when you use `$this->validate()`, but you can also load it manually:  

```php
$validation = \Config\Services::validation();
```

Alternatively, you can **autoload** it in `app/Config/Autoload.php`:  

```php
public $libraries = ['validation'];
```

---

## **2. Creating a Form with Validation**  

### **A. Controller with Validation Rules**  
📁 `app/Controllers/FormController.php`  

```php
<?php
namespace App\Controllers;

use CodeIgniter\Controller;

class FormController extends Controller
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

### **B. Creating the Form View**  
📁 `app/Views/form_view.php`  

```php
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Form Validation in CodeIgniter</title>
</head>
<body>
    <h2>Form Validation Example</h2>

    <?php if (isset($validation)) : ?>
        <div style="color: red;">
            <?= $validation->listErrors(); ?>
        </div>
    <?php endif; ?>

    <form action="<?= site_url('formcontroller/submit') ?>" method="post">
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

✅ **This view will display validation errors and the form inputs!**  

---

## **3. Customizing Error Messages**  
Instead of default messages, you can **define custom error messages** in the controller:  

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

✅ **This makes error messages more readable for users!**  

---

## **4. Validating Checkbox, Radio, and Dropdown Fields**  

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

✅ **These rules ensure that users make valid selections!**  

---

## **5. Redirecting with Flash Messages (Success or Error)**  
Instead of showing errors directly, you can **redirect users with flash messages**:  

📁 **Modify `submit()` in `FormController.php`**  
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

✅ **Now users see feedback messages even after redirection!**  

---

## **6. Client-Side Validation (JavaScript)**  
While CodeIgniter ensures server-side validation, adding JavaScript prevents unnecessary form submissions.  

📁 **Modify `form_view.php`**  
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
✅ **This prevents users from submitting incomplete forms!**  

---

## **7. Database Validation (Checking Unique Values)**  
To ensure an **email is unique**, modify validation rules:  

```php
'email' => 'required|valid_email|is_unique[users.email]'
```

📌 This checks if the email **already exists** in the `users` table.  

✅ **Prevents duplicate entries in the database!**  

---

## **Final Thoughts**  
✔ **CodeIgniter’s Form Validation Library** simplifies validation.  
✔ **Use server-side validation** to ensure security.  
✔ **Enhance UX with JavaScript validation**.  
✔ **Store errors in session flashdata** for better user experience.  
✔ **Use `is_unique` rule** for database validation.  
